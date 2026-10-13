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
    unresolved: dict[str, str] = {}
    for hyp in targets:
        for iteration in range(1, max_iters + 1):
            bb.log("analyze", "iteration", f"{hyp.id}: {iteration}/{max_iters}")
            folder = workdir / "hypotheses" / hyp.id
            try:
                await agentic_analyst.run(bb, hyp, workdir=workdir, model_id=analyst_model,
                                          background_model=background_model,
                                          usage_log=getattr(client, "usage_log", None))
            except agentic_analyst.NotTestable as exc:
                _record_outcome(bb, hyp, folder, "not_testable",
                                f"the analyst found: {exc}", not_testable)
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
            limits = [i for i in high if i.get("kind") == "data_limit"]
            last_round = iteration == max_iters
            if limits and (last_round or len(limits) == len(high)):
                # No revision of the analysis can fix a data limit. Before the
                # last round the loop only stops when every high-severity issue
                # is one, so a fixable issue still gets its revision. On the last
                # round there are no revisions left, so a remaining data limit is
                # the binding reason the analysis did not pass.
                reasons = "; ".join(i.get("description", "") for i in limits)
                _record_outcome(bb, hyp, folder, "not_testable",
                                f"the methods critic found: {reasons}", not_testable)
                break
            if last_round:
                # The review never passed, but the rounds happened and the
                # notebooks are real. That is a result to report, not a crash:
                # the stage still does not pass, so Write builds nothing on it.
                reasons = "; ".join(i.get("description", "") for i in high)
                detail = (f"still found: {reasons}" if reasons else
                          f"did not pass the analysis (verdict: {critique.get('verdict', 'unknown')})")
                _record_outcome(bb, hyp, folder, "unresolved",
                                f"after {max_iters} rounds the methods critic {detail}", unresolved)
                break
            bb.log("analyze", "revision_required", f"{hyp.id}: {json.dumps(critique)}")
    if not passed and unresolved:
        raise ReviewUnresolved(unresolved, not_testable)
    if not passed and not_testable:
        raise NotTestableStage(not_testable)


class NotTestableStage(RuntimeError):
    """No analyzed hypothesis passed, and at least one cannot be tested with the
    deposit. The stage stops, without a manuscript, and says why."""

    def __init__(self, reasons: dict[str, str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(f"{h} is not testable with this deposit: {r}" for h, r in reasons.items()))


class ReviewUnresolved(RuntimeError):
    """No analyzed hypothesis passed its methods review within the rounds
    allowed. The stage stops, without a manuscript, and says what was left.

    This is the honest end of a review that ran out of rounds. It used to be a
    bare `RuntimeError` raised from the loop, which threw away three rounds of
    real work behind a traceback and told Write nothing.
    """

    def __init__(self, unresolved: dict[str, str], not_testable: dict[str, str] | None = None) -> None:
        self.unresolved = unresolved
        self.not_testable = not_testable or {}
        parts = [f"{h}: {r}" for h, r in unresolved.items()]
        parts += [f"{h} is not testable with this deposit: {r}" for h, r in self.not_testable.items()]
        super().__init__("; ".join(parts))


def _record_outcome(bb: Blackboard, hyp, folder: Path, outcome: str, reason: str,
                    out: dict[str, str]) -> None:
    critique = dict(hyp.methods_critique or {})
    critique.update(outcome=outcome, not_testable_reason=reason)
    hyp.methods_critique = critique
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{outcome}.json").write_text(
        json.dumps({"hypothesis_id": hyp.id, "outcome": outcome, "reason": reason}, indent=2))
    bb.log("analyze", outcome, f"{hyp.id}: {reason}")
    out[hyp.id] = reason
