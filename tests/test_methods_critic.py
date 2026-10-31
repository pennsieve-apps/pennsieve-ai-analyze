from __future__ import annotations

import pytest

from analyze.agents import methods_critic
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis


def test_critic_failure_propagates_after_logging(monkeypatch):
    bb = Blackboard(research_question="question", subsystem="dataset")
    hypothesis = Hypothesis(
        id="H1",
        claim="claim",
        rationale="rationale",
        supporting_evidence=[],
        predicted_observation="observation",
        suggested_experiment="experiment",
        analysis_summary="analysis completed",
    )

    def fail_call(*args, **kwargs):
        raise RuntimeError("governor unavailable")

    monkeypatch.setattr(methods_critic, "call_json", fail_call)
    monkeypatch.setattr(methods_critic, "_read_notebook_text", lambda h: '{"cells": []}')

    with pytest.raises(RuntimeError, match="Methods critique failed"):
        methods_critic.run(bb, object(), hypothesis)

    assert bb.notebook[-1]["agent"] == "methods_critic"
    assert bb.notebook[-1]["action"] == "error"


def test_the_critic_judges_the_analysis_against_the_approved_plan(monkeypatch, tmp_path):
    """It cannot catch a dropped arm without being told what was approved."""
    import json as _json
    from analyze.agents import methods_critic as mc

    notebook = tmp_path / "analysis.ipynb"
    notebook.write_text(_json.dumps({"cells": [
        {"cell_type": "code", "source": "print(1)", "execution_count": 1, "outputs": []}]}))
    (tmp_path / "analysis_plan.json").write_text(_json.dumps(
        {"primary_comparison": "H15K6 vs CD227 and H4K11", "arms": ["H15K6", "CD227", "H4K11"],
         "metric": "background-subtracted gMFI", "statistical_test": "paired",
         "deviations": [{"changed": "dropped H4K11", "why": "below the floor"}]}))
    seen = {}

    def fake_call_json(client, **kwargs):
        seen.update(kwargs)
        return {"verdict": "sound", "issues": [], "suggested_followups": []}

    monkeypatch.setattr(mc, "call_json", fake_call_json)
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [],
                   "H15K6 falls more than CD227 and H4K11", "stain all three clones")
    h.analysis_summary = "summary"
    mc.run(bb, object(), h, notebook_path=notebook)

    assert "H15K6 falls more than CD227 and H4K11" in seen["user"]
    assert "stain all three clones" in seen["user"]
    assert "dropped H4K11" in seen["user"], "the declared deviation is put to the critic"
    assert "plan_deviation" in seen["system"]
