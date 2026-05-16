"""Orchestration for the analyze stage.

For each hypothesis on the blackboard, run the agentic analyst (writes
+ executes a notebook against the named dataset(s)) and the methods
critic (reviews the notebook + outputs and flags failed assumptions /
weak inferences). Results are recorded back onto the hypothesis.

v0 scope: single pass, top-N hypotheses (configurable via env var,
default 1). The HypGen ↔ Critic loop is upstream in Hypothesize; here
we trust the ranking and analyze the highest-ranked candidates.
"""
from __future__ import annotations

import os
from pathlib import Path

from analyze.blackboard import Blackboard
from analyze.llm import LLMClient
from analyze.agents import agentic_analyst, methods_critic


def _hypotheses_to_analyze(bb: Blackboard) -> list:
    """Pick which hypotheses get analyzed in this run.

    Default: the top hypothesis by rank (or first in list if unranked).
    Override via `VP_ANALYZE_TOP_N` env var.
    """
    n = int(os.environ.get("VP_ANALYZE_TOP_N", "1"))
    return list(bb.hypotheses)[:n]


async def run_analyze(bb: Blackboard, client: LLMClient, *, workdir: Path) -> None:
    """Top-level analyze pipeline. Async because the agentic analyst uses
    the Claude Agent SDK, which is async."""
    targets = _hypotheses_to_analyze(bb)
    if not targets:
        bb.log("analyze", "no_hypotheses",
               "blackboard has no hypotheses to analyze — Hypothesize "
               "produced an empty output; nothing to do.")
        return

    print(f"\n--- agentic analyst (analyzing {len(targets)} hypothesis/es) ---")
    for hyp in targets:
        await agentic_analyst.run(bb, hyp, workdir=workdir)

    print("\n--- methods critic ---")
    for hyp in targets:
        methods_critic.run(bb, client, hyp)
