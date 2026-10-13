"""Orchestration for the analyze stage.

For each hypothesis on the blackboard, run the agentic analyst (writes
+ executes a notebook against the named dataset(s)) and the methods
critic (reviews the notebook + outputs and flags failed assumptions /
weak inferences). Results are recorded back onto the hypothesis.

Analyze up to N accepted hypotheses (default 1), with a bounded methods
revision loop. Rejected or unreviewed candidates cannot enter analysis.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from analyze.agents import agentic_analyst, methods_critic
from pennsieve_ai_utils.blackboard import Blackboard
from pennsieve_ai_utils.llm import LLMClient


def _hypotheses_to_analyze(bb: Blackboard) -> list:
    """Pick which hypotheses get analyzed in this run.

    Default: the first accepted hypothesis in Hypothesize's final iteration. Locked
    hypotheses keep their IDs across iterations, so selecting from the final
    slate ensures Write later reads the same record that Analyze updates.
    Override the count via `VP_ANALYZE_TOP_N`.
    """
    n = int(os.environ.get("VP_ANALYZE_TOP_N", "1"))
    if n < 1:
        raise ValueError("VP_ANALYZE_TOP_N must be at least 1")
    if not bb.hypotheses:
        return []
    final_iteration = bb.iterations_completed or max(h.iteration for h in bb.hypotheses)
    reviews = {r.hypothesis_id: r for r in bb.reviews_at(final_iteration)}
    return [h for h in bb.hypotheses_at(final_iteration)
            if h.id in reviews and reviews[h.id].verdict == "accept"][:n]


async def run_analyze(bb: Blackboard, client: LLMClient, *, workdir: Path) -> None:
    """Top-level analyze pipeline. Async because the agentic analyst uses
    the Claude Agent SDK, which is async."""
    targets = _hypotheses_to_analyze(bb)
    if not targets:
        message = ("No accepted hypotheses in the final iteration. Review the "
                   "hypothesis critiques; analysis and manuscript generation are stopped.")
        bb.log("analyze", "blocked", message)
        raise RuntimeError(message)

    max_iters = int(os.environ.get("VP_ANALYZE_MAX_METHODS_ITERS", "3"))
    if max_iters < 1:
        raise ValueError("VP_ANALYZE_MAX_METHODS_ITERS must be at least 1")

    print(f"\n--- agentic analyst (analyzing {len(targets)} hypothesis/es) ---")
    analyst_model = client.resolve_model("sonnet")
    # Claude Code's background calls need a model this account can invoke
    # when it talks to Bedrock or the Anthropic API directly.
    background_model = ("" if getattr(client, "backend", "governor") == "governor"
                        else client.resolve_model("bulk"))
    passed: list[str] = []
    not_testable: dict[str, str] = {}
    for hyp in targets:
        for iteration in range(1, max_iters + 1):
            bb.log("analyze", "iteration", f"{hyp.id}: {iteration}/{max_iters}")
            folder = workdir / "hypotheses" / hyp.id
            try:
                await agentic_analyst.run(bb, hyp, workdir=workdir, model_id=analyst_model,
                                          background_model=background_model,
                                          usage_log=getattr(client, "usage_log", None))
            except agentic_analyst.NotTestable as exc:
                _record_not_testable(bb, hyp, folder, f"the analyst found: {exc}", not_testable)
                break
            methods_critic.run(bb, client, hyp, notebook_path=folder / "analysis.ipynb")
            critique = hyp.methods_critique
            (folder / f"methods.iter{iteration}.json").write_text(json.dumps(critique, indent=2))
            shutil.copy2(folder / "analysis.ipynb", folder / f"analysis.iter{iteration}.ipynb")
            bb.save(workdir / f"blackboard.{hyp.id}.iter{iteration}.json")
            high = [i for i in critique.get("issues", []) if i.get("severity") == "high"]
            if critique.get("verdict") == "sound" and not high:
                passed.append(hyp.id)
                break
            if high and all(i.get("kind") == "data_limit" for i in high):
                # No revision of the analysis can fix these, so another round would
                # only repeat them. Record the honest outcome and stop.
                reasons = "; ".join(i.get("description", "") for i in high)
                _record_not_testable(bb, hyp, folder, f"the methods critic found: {reasons}", not_testable)
                break
            bb.log("analyze", "revision_required", f"{hyp.id}: {json.dumps(critique)}")
        else:
            raise RuntimeError(f"{hyp.id}: methods review did not pass after {max_iters} iterations; "
                               "notebook and critiques retained for review")
    if not passed and not_testable:
        raise NotTestableStage(not_testable)


class NotTestableStage(RuntimeError):
    """No analyzed hypothesis passed, and at least one cannot be tested with the
    deposit. The stage stops, without a manuscript, and says why."""

    def __init__(self, reasons: dict[str, str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(f"{h} is not testable with this deposit: {r}" for h, r in reasons.items()))


def _record_not_testable(bb: Blackboard, hyp, folder: Path, reason: str, out: dict[str, str]) -> None:
    critique = dict(hyp.methods_critique or {})
    critique.update(outcome="not_testable", not_testable_reason=reason)
    hyp.methods_critique = critique
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "not_testable.json").write_text(json.dumps({"hypothesis_id": hyp.id, "reason": reason}, indent=2))
    bb.log("analyze", "not_testable", f"{hyp.id}: {reason}")
    out[hyp.id] = reason
