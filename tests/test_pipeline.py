from __future__ import annotations

import asyncio
import pytest

from analyze import pipeline
from analyze.blackboard import Blackboard, Hypothesis, CriticReview


def _review(iteration, verdict="accept"):
    return CriticReview("H1", 4, 4, 4, 4, [], verdict, iteration)


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
        critic_reviews=[_review(3)],
    )
    analyst_calls = []
    critic_calls = []

    async def fake_analyst(bb, hypothesis, *, workdir, model_id):
        analyst_calls.append((hypothesis, workdir, model_id))
        hypothesis.analysis_workdir = "hypotheses/H1"
        hypothesis.analysis_summary = "complete"
        folder = workdir / "hypotheses" / hypothesis.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.ipynb").write_text("{}")

    def fake_critic(bb, client, hypothesis, **kwargs):
        critic_calls.append(hypothesis)
        hypothesis.methods_critique = {"verdict": "sound", "issues": []}

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
        critic_reviews=[_review(2)],
    )
    assert pipeline._hypotheses_to_analyze(bb) == [final]


def test_rejected_or_stale_acceptance_cannot_start_analysis(tmp_path):
    bb = Blackboard("q", "dataset", hypotheses=[_hypothesis(2)],
                    critic_reviews=[_review(1), _review(2, "reject")])
    with pytest.raises(RuntimeError, match="No accepted hypotheses"):
        asyncio.run(pipeline.run_analyze(bb, object(), workdir=tmp_path))


@pytest.mark.parametrize("pass_second", [True, False])
def test_methods_revision_loop_is_bounded_and_snapshotted(monkeypatch, tmp_path, pass_second):
    monkeypatch.setenv("VP_ANALYZE_MAX_METHODS_ITERS", "2")
    h = _hypothesis(1)
    bb = Blackboard("q", "dataset", hypotheses=[h], critic_reviews=[_review(1)])
    seen = []
    async def analyst(bb, hyp, *, workdir, model_id):
        seen.append(dict(hyp.methods_critique))
        folder = workdir / "hypotheses" / hyp.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.ipynb").write_text(str(len(seen)))
    def critic(bb, client, hyp, **kwargs):
        hyp.methods_critique = {"verdict": "sound" if pass_second and len(seen) == 2 else "concerns", "issues": []}
    monkeypatch.setattr(pipeline.agentic_analyst, "run", analyst)
    monkeypatch.setattr(pipeline.methods_critic, "run", critic)
    client = type("Client", (), {"resolve_model": lambda self, role: "model"})()
    if pass_second:
        asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    else:
        with pytest.raises(RuntimeError, match="did not pass"):
            asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert len(seen) == 2 and seen[1]["verdict"] == "concerns"
    assert (tmp_path / "hypotheses/H1/analysis.iter1.ipynb").read_text() == "1"
    assert (tmp_path / "hypotheses/H1/analysis.iter2.ipynb").read_text() == "2"
