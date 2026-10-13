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
    for hyp in targets:
        for iteration in range(1, max_iters + 1):
            bb.log("analyze", "iteration", f"{hyp.id}: {iteration}/{max_iters}")
            await agentic_analyst.run(bb, hyp, workdir=workdir, model_id=analyst_model,
                                      background_model=background_model,
                                      usage_log=getattr(client, "usage_log", None))
            folder = workdir / "hypotheses" / hyp.id
            methods_critic.run(bb, client, hyp, notebook_path=folder / "analysis.ipynb")
            critique = hyp.methods_critique
            (folder / f"methods.iter{iteration}.json").write_text(json.dumps(critique, indent=2))
            shutil.copy2(folder / "analysis.ipynb", folder / f"analysis.iter{iteration}.ipynb")
            bb.save(workdir / f"blackboard.{hyp.id}.iter{iteration}.json")
            if critique.get("verdict") == "sound" and not any(
                issue.get("severity") == "high" for issue in critique.get("issues", [])
            ):
                break
            bb.log("analyze", "revision_required", f"{hyp.id}: {json.dumps(critique)}")
        else:
            raise RuntimeError(f"{hyp.id}: methods review did not pass after {max_iters} iterations; "
                               "notebook and critiques retained for review")
