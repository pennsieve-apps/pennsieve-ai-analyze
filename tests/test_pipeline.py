from __future__ import annotations

import asyncio

from analyze import pipeline
from analyze.blackboard import Blackboard, Hypothesis


def _hypothesis(iteration: int) -> Hypothesis:
    return Hypothesis(
        id="H1",
        claim=f"iteration {iteration}",
        rationale="rationale",
        supporting_evidence=[],
        predicted_observation="observation",
        suggested_experiment="experiment",
        iteration=iteration,
    )


def test_analyze_targets_final_iteration_hypothesis(monkeypatch, tmp_path):
    first = _hypothesis(1)
    final = _hypothesis(3)
    bb = Blackboard(
        research_question="question",
        subsystem="pennsieve:530",
        iterations_completed=3,
        hypotheses=[first, final],
    )
    analyst_calls = []
    critic_calls = []

    async def fake_analyst(bb, hypothesis, *, workdir, model_id):
        analyst_calls.append((hypothesis, workdir, model_id))
        hypothesis.analysis_workdir = "hypotheses/H1"
        hypothesis.analysis_summary = "complete"

    def fake_critic(bb, client, hypothesis):
        critic_calls.append(hypothesis)

    monkeypatch.setattr(pipeline.agentic_analyst, "run", fake_analyst)
    monkeypatch.setattr(pipeline.methods_critic, "run", fake_critic)
    client = type(
        "Client", (),
        {"resolve_model": lambda self, role: "us.anthropic.claude-sonnet-4-6"},
    )()

    asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))

    assert analyst_calls == [(final, tmp_path, "us.anthropic.claude-sonnet-4-6")]
    assert critic_calls == [final]
    assert first.analysis_summary == ""
    assert final.analysis_summary == "complete"


def test_selection_falls_back_to_highest_iteration(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    first = _hypothesis(1)
    final = _hypothesis(2)
    bb = Blackboard(
        research_question="question", subsystem="dataset",
        hypotheses=[first, final],
    )
    assert pipeline._hypotheses_to_analyze(bb) == [final]
