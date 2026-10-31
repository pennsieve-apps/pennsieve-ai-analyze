from __future__ import annotations

import asyncio
import json
import pytest

from analyze import pipeline
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis, CriticReview


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
    background_models = []
    critic_calls = []

    async def fake_analyst(bb, hypothesis, *, workdir, model_id, background_model="", usage_log=None):
        analyst_calls.append((hypothesis, workdir, model_id))
        background_models.append(background_model)
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
    # The governor path leaves Claude Code's background model to the platform.
    assert background_models == [""]
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
    async def analyst(bb, hyp, *, workdir, model_id, background_model="", usage_log=None):
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
        with pytest.raises(pipeline.ReviewUnresolved, match="did not pass the analysis"):
            asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert len(seen) == 2 and seen[1]["verdict"] == "concerns"
    assert (tmp_path / "hypotheses/H1/analysis.iter1.ipynb").read_text() == "1"
    assert (tmp_path / "hypotheses/H1/analysis.iter2.ipynb").read_text() == "2"


def test_direct_backend_passes_a_background_model(monkeypatch, tmp_path):
    hyp = _hypothesis(1)
    bb = Blackboard(research_question="question", subsystem="pennsieve:530",
                    iterations_completed=1, hypotheses=[hyp], critic_reviews=[_review(1)])
    seen = []

    async def fake_analyst(bb, hypothesis, *, workdir, model_id, background_model="", usage_log=None):
        seen.append(background_model)
        folder = workdir / "hypotheses" / hypothesis.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.ipynb").write_text("{}")

    def fake_critic(bb, client, hypothesis, **kwargs):
        hypothesis.methods_critique = {"verdict": "sound", "issues": []}

    monkeypatch.setattr(pipeline.agentic_analyst, "run", fake_analyst)
    monkeypatch.setattr(pipeline.methods_critic, "run", fake_critic)
    models = {"sonnet": "us.anthropic.claude-sonnet-5-5",
              "bulk": "us.anthropic.claude-haiku-4-5-20251001-v1:0"}
    client = type("Client", (), {"backend": "bedrock",
                                 "resolve_model": lambda self, role: models[role]})()

    asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))

    assert seen == ["us.anthropic.claude-haiku-4-5-20251001-v1:0"]


def _loop(monkeypatch, tmp_path, issues, analyst_raises=None):
    """Run the methods loop with a fake analyst and a critic that always returns `issues`."""
    monkeypatch.setenv("VP_ANALYZE_MAX_METHODS_ITERS", "3")
    h = _hypothesis(1)
    bb = Blackboard("q", "dataset", hypotheses=[h], critic_reviews=[_review(1)])
    rounds = []

    async def analyst(bb, hyp, *, workdir, model_id, background_model="", usage_log=None):
        rounds.append(1)
        folder = workdir / "hypotheses" / hyp.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.ipynb").write_text("{}")
        if analyst_raises:
            raise analyst_raises

    def critic(bb, client, hyp, **kwargs):
        hyp.methods_critique = {"verdict": "concerns", "issues": issues}
    monkeypatch.setattr(pipeline.agentic_analyst, "run", analyst)
    monkeypatch.setattr(pipeline.methods_critic, "run", critic)
    client = type("Client", (), {"resolve_model": lambda self, role: "model"})()
    return bb, h, rounds, client


def test_only_data_limits_stop_the_loop_as_not_testable(monkeypatch, tmp_path):
    issues = [{"severity": "high", "kind": "data_limit", "description": "staining panel is not an ADCP assay"},
              {"severity": "medium", "kind": "fixable", "description": "no doublet gate"}]
    bb, h, rounds, client = _loop(monkeypatch, tmp_path, issues)
    with pytest.raises(pipeline.NotTestableStage, match="not testable with this deposit: the methods critic"):
        asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert len(rounds) == 1, "no further rounds once only data limits remain"
    assert h.methods_critique["outcome"] == "not_testable"
    assert "staining panel" in (tmp_path / "hypotheses/H1/not_testable.json").read_text()


def test_a_fixable_high_issue_keeps_the_loop_going(monkeypatch, tmp_path):
    """A data limit beside a fixable issue still gets its revisions, and once the
    rounds run out the data limit is the reason the analysis did not pass."""
    issues = [{"severity": "high", "kind": "data_limit", "description": "no replicates"},
              {"severity": "high", "kind": "fixable", "description": "uncompensated channels"}]
    bb, h, rounds, client = _loop(monkeypatch, tmp_path, issues)
    with pytest.raises(pipeline.NotTestableStage, match="no replicates"):
        asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert len(rounds) == 3, "the fixable issue gets every round it is owed"
    assert h.methods_critique["outcome"] == "not_testable"
    # The reason names the data limit, not the issue a revision could have fixed.
    assert "uncompensated" not in (tmp_path / "hypotheses/H1/not_testable.json").read_text()


def test_a_review_that_never_passes_is_recorded_not_raised(monkeypatch, tmp_path):
    """Running out of rounds with only fixable issues is a reported outcome.

    It used to raise a bare RuntimeError, which threw away three rounds of real
    work behind a traceback and told Write nothing.
    """
    issues = [{"severity": "high", "kind": "fixable", "description": "uncompensated channels"}]
    bb, h, rounds, client = _loop(monkeypatch, tmp_path, issues)
    with pytest.raises(pipeline.ReviewUnresolved, match="uncompensated channels"):
        asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert len(rounds) == 3
    assert h.methods_critique["outcome"] == "unresolved"
    recorded = json.loads((tmp_path / "hypotheses/H1/unresolved.json").read_text())
    assert recorded["outcome"] == "unresolved" and "uncompensated" in recorded["reason"]


def test_an_unresolved_review_is_not_a_pass(monkeypatch, tmp_path):
    """The hypothesis must not look analyzed to the supervisor's gate."""
    issues = [{"severity": "high", "kind": "fixable", "description": "uncompensated channels"}]
    bb, h, rounds, client = _loop(monkeypatch, tmp_path, issues)
    with pytest.raises(pipeline.ReviewUnresolved):
        asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert h.methods_critique["verdict"] != "sound"
    assert any(i["severity"] == "high" for i in h.methods_critique["issues"])


def test_the_analyst_reporting_not_testable_skips_the_critic(monkeypatch, tmp_path):
    bb, h, rounds, client = _loop(monkeypatch, tmp_path, [],
                                  analyst_raises=pipeline.agentic_analyst.NotTestable("only 3 clones have ADCP data"))
    with pytest.raises(pipeline.NotTestableStage, match="the analyst found: only 3 clones"):
        asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    assert len(rounds) == 1 and h.methods_critique["outcome"] == "not_testable"


def _hyp(hid, iteration=3):
    return Hypothesis(id=hid, claim=hid, rationale="r", supporting_evidence=[],
                      predicted_observation="p", suggested_experiment="e", iteration=iteration)


def _accept(hid, iteration=3):
    return CriticReview(hid, 4, 4, 4, 4, [], "accept", iteration)


def test_a_named_hypothesis_id_is_selected(monkeypatch):
    """VP_ANALYZE_HYPOTHESIS_IDS picks a specific accepted hypothesis when the
    first one turns out not to be testable."""
    monkeypatch.setenv("VP_ANALYZE_HYPOTHESIS_IDS", "H3")
    bb = Blackboard("q", "pennsieve:405", iterations_completed=3,
                    hypotheses=[_hyp("H2"), _hyp("H3"), _hyp("H5")],
                    critic_reviews=[_accept("H2"), _accept("H3"), _accept("H5")])
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H3"]


def test_named_ids_keep_their_order(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_HYPOTHESIS_IDS", "H5, H2")
    bb = Blackboard("q", "s", iterations_completed=3,
                    hypotheses=[_hyp("H2"), _hyp("H3"), _hyp("H5")],
                    critic_reviews=[_accept("H2"), _accept("H3"), _accept("H5")])
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H5", "H2"]


def test_a_named_id_that_was_not_accepted_is_refused(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_HYPOTHESIS_IDS", "H4")
    bb = Blackboard("q", "s", iterations_completed=3,
                    hypotheses=[_hyp("H2"), _hyp("H3")],
                    critic_reviews=[_accept("H2"), _accept("H3")])
    with pytest.raises(ValueError, match="H4.*not accepted"):
        pipeline._hypotheses_to_analyze(bb)


def _scored(hid, novelty, plausibility, feasibility, alignment, attacks=(), iteration=3):
    return CriticReview(hid, novelty, plausibility, feasibility, alignment,
                        list(attacks), "accept", iteration)


def test_the_best_scored_hypothesis_is_chosen_not_the_first_listed(monkeypatch):
    """The final scores from run seed530-write: list order picked H4, but H6
    ties it on novelty, plausibility and alignment and is more feasible."""
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    ids = ["H4", "H1", "H2", "H3", "H6"]
    bb = Blackboard("q", "pennsieve:530", iterations_completed=3,
                    hypotheses=[_hyp(h) for h in ids],
                    critic_reviews=[_scored("H4", 4, 3, 3, 5), _scored("H1", 3, 4, 3, 5),
                                    _scored("H2", 4, 3, 3, 4), _scored("H3", 3, 3, 4, 4),
                                    _scored("H6", 4, 3, 4, 5)])
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H6"]
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "5")
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H6", "H3", "H4", "H2", "H1"]
    assert "chose H6, H3, H4, H2, H1 of 5 accepted" in bb.notebook[-1]["content"]


def test_fewer_blocking_attacks_break_a_score_tie(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    bb = Blackboard("q", "s", iterations_completed=3,
                    hypotheses=[_hyp("H1"), _hyp("H2")],
                    critic_reviews=[_scored("H1", 4, 4, 4, 4, ["[READOUT] no such column", "[REASONING] x"]),
                                    _scored("H2", 4, 4, 4, 4, ["[REASONING] y", "[REASONING] z"])])
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H2"]


def test_list_order_breaks_a_full_tie(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "2")
    bb = Blackboard("q", "s", iterations_completed=3,
                    hypotheses=[_hyp("H2"), _hyp("H3"), _hyp("H5")],
                    critic_reviews=[_accept("H2"), _accept("H3"), _accept("H5")])
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H2", "H3"]


def test_named_ids_override_the_ranking(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_HYPOTHESIS_IDS", "H1")
    bb = Blackboard("q", "s", iterations_completed=3,
                    hypotheses=[_hyp("H1"), _hyp("H2")],
                    critic_reviews=[_scored("H1", 1, 1, 1, 1), _scored("H2", 5, 5, 5, 5)])
    assert [h.id for h in pipeline._hypotheses_to_analyze(bb)] == ["H1"]


def _ranked_board():
    """H2 ranks first, then H1, then H3."""
    return Blackboard("q", "s", iterations_completed=3,
                      hypotheses=[_hyp("H1"), _hyp("H2"), _hyp("H3")],
                      critic_reviews=[_scored("H1", 4, 4, 3, 4), _scored("H2", 4, 4, 4, 4),
                                      _scored("H3", 3, 3, 3, 3)])


def _run_with(monkeypatch, tmp_path, bb, untestable=()):
    """Run Analyze with an analyst that reports each ID in `untestable` as not
    testable and a critic that passes everything else. Returns the IDs tried."""
    tried = []

    async def analyst(bb, hyp, *, workdir, model_id, background_model="", usage_log=None):
        tried.append(hyp.id)
        folder = workdir / "hypotheses" / hyp.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.ipynb").write_text("{}")
        if hyp.id in untestable:
            raise pipeline.agentic_analyst.NotTestable("no such measurement")

    def critic(bb, client, hyp, **kwargs):
        hyp.methods_critique = {"verdict": "sound", "issues": []}
    monkeypatch.setattr(pipeline.agentic_analyst, "run", analyst)
    monkeypatch.setattr(pipeline.methods_critic, "run", critic)
    client = type("Client", (), {"resolve_model": lambda self, role: "model"})()
    asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    return tried


def test_a_failed_hypothesis_falls_back_to_the_next_ranked(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    bb = _ranked_board()
    assert _run_with(monkeypatch, tmp_path, bb, untestable={"H2"}) == ["H2", "H1"]
    assert any(e["action"] == "fallback" and e["content"].startswith("H1") for e in bb.notebook)
    assert "then H1 if one fails" in next(e["content"] for e in bb.notebook if e["action"] == "selection")


def test_no_fallback_runs_once_enough_have_passed(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    bb = _ranked_board()
    assert _run_with(monkeypatch, tmp_path, bb) == ["H2"]
    assert not any(e["action"] == "fallback" for e in bb.notebook)


def test_max_attempts_caps_the_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    monkeypatch.setenv("VP_ANALYZE_MAX_ATTEMPTS", "1")
    with pytest.raises(pipeline.NotTestableStage, match="H2"):
        _run_with(monkeypatch, tmp_path, _ranked_board(), untestable={"H2"})


def test_max_attempts_below_top_n_is_refused(monkeypatch):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "2")
    monkeypatch.setenv("VP_ANALYZE_MAX_ATTEMPTS", "1")
    with pytest.raises(ValueError, match="at least VP_ANALYZE_TOP_N"):
        pipeline._hypotheses_to_analyze(_ranked_board())


def test_named_ids_get_no_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_ANALYZE_HYPOTHESIS_IDS", "H3")
    with pytest.raises(pipeline.NotTestableStage, match="H3"):
        _run_with(monkeypatch, tmp_path, _ranked_board(), untestable={"H3"})


def _refusing_run(monkeypatch, tmp_path, bb, refusing_models):
    """Run Analyze with an analyst that refuses on `refusing_models`. Returns
    (hypothesis ID, model) for each analyst call."""
    calls = []

    async def analyst(bb, hyp, *, workdir, model_id, background_model="", usage_log=None):
        calls.append((hyp.id, model_id))
        if model_id in refusing_models:
            raise pipeline.agentic_analyst.AnalystRefused(model_id)
        folder = workdir / "hypotheses" / hyp.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "analysis.ipynb").write_text("{}")
        hyp.analysis_workdir, hyp.analysis_summary = f"hypotheses/{hyp.id}", "done"

    def critic(bb, client, hyp, **kwargs):
        hyp.methods_critique = {"verdict": "sound", "issues": []}
    monkeypatch.setattr(pipeline.agentic_analyst, "run", analyst)
    monkeypatch.setattr(pipeline.methods_critic, "run", critic)
    client = type("Client", (), {"resolve_model": lambda self, role: "opus-5-5", "usage_log": []})()
    asyncio.run(pipeline.run_analyze(bb, client, workdir=tmp_path))
    return calls, client.usage_log


def test_a_refused_round_reruns_on_the_fallback_and_stays_there(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "2")
    monkeypatch.setenv("VP_ANALYZE_FALLBACK_MODEL", "opus-4-7")
    calls, usage = _refusing_run(monkeypatch, tmp_path, _ranked_board(), {"opus-5-5"})
    assert calls == [("H2", "opus-5-5"), ("H2", "opus-4-7"), ("H1", "opus-4-7")]
    assert usage == [{"event": "refusal", "agent": "agentic_analyst", "hypothesis_id": "H2",
                      "model_id": "opus-5-5", "fallback_model_id": "opus-4-7"}]


def test_when_every_model_refuses_the_next_hypothesis_is_tried(monkeypatch, tmp_path):
    monkeypatch.setenv("VP_ANALYZE_TOP_N", "1")
    monkeypatch.setenv("VP_ANALYZE_FALLBACK_MODEL", "none")
    with pytest.raises(pipeline.AllAnalystsRefused, match="H2: every analyst model refused"):
        _refusing_run(monkeypatch, tmp_path, _ranked_board(), {"opus-5-5"})
