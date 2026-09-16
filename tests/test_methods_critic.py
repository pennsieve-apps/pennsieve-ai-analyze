from __future__ import annotations

import pytest

from analyze.agents import methods_critic
from analyze.blackboard import Blackboard, Hypothesis


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
