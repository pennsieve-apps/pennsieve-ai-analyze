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


# Critic attack tags saying the lab or the deposit cannot support the test as
# described. Fewer of them breaks ties between equally scored hypotheses.
BLOCKING_ATTACK_TAGS = ("[CONSTRAINT]", "[READOUT]")


def _rank_key(review, position: int) -> tuple:
    """Sort key for an accepted hypothesis: best first.

    Feasibility leads because a limited budget is wasted on a hypothesis the
    data cannot test. Then novelty, then the sum of all four critic scores,
    then fewer blocking attacks. Hypothesize's list order breaks what is left.
    """
    total = review.novelty + review.plausibility + review.feasibility + review.alignment
    blocking = sum(1 for a in review.attacks if a.lstrip().startswith(BLOCKING_ATTACK_TAGS))
    return (-review.feasibility, -review.novelty, -total, blocking, position)


def _hypotheses_to_analyze(bb: Blackboard) -> list:
    """The hypotheses this run means to pass: the first `wanted` of the queue."""
    queue, wanted = _analysis_queue(bb)
    return queue[:wanted]


def _analysis_queue(bb: Blackboard) -> tuple[list, int]:
    """Order the hypotheses Analyze may try, and say how many should pass.

    Default: accepted hypotheses in Hypothesize's final iteration, best first,
    ranked by the critic's scores (see `_rank_key`). Locked hypotheses keep
    their IDs across iterations, so selecting from the final slate ensures
    Write later reads the same record that Analyze updates.

    `VP_ANALYZE_TOP_N` sets how many should pass; the default, `all`, tests
    every accepted hypothesis. With a number, when one turns out not
    testable or its methods review stays unresolved, the run moves on to the
    next-ranked hypothesis, up to `VP_ANALYZE_MAX_ATTEMPTS` tried in all
    (default one more than `VP_ANALYZE_TOP_N`). That cap bounds the spending.

    `VP_ANALYZE_HYPOTHESIS_IDS`, a comma-separated list, names exactly which
    ones instead, in the order given, with no fallback. Each named ID must be
    accepted at the final iteration, because Analyze only runs what Hypothesize
    passed and Write only writes what passes methods review.
    """
    if not bb.hypotheses:
        return [], 0
    final_iteration = bb.iterations_completed or max(h.iteration for h in bb.hypotheses)
    reviews = {r.hypothesis_id: r for r in bb.reviews_at(final_iteration)}
    accepted = [h for h in bb.hypotheses_at(final_iteration)
                if h.id in reviews and reviews[h.id].verdict == "accept"]

    wanted = [p.strip() for p in os.environ.get("VP_ANALYZE_HYPOTHESIS_IDS", "").split(",") if p.strip()]
    if wanted:
        by_id = {h.id: h for h in accepted}
        missing = [hid for hid in wanted if hid not in by_id]
        if missing:
            have = ", ".join(by_id) or "(none)"
            raise ValueError(
                f"VP_ANALYZE_HYPOTHESIS_IDS names {', '.join(missing)}, "
                f"which {'is' if len(missing) == 1 else 'are'} not accepted at the final "
                f"iteration. Accepted and analyzable: {have}.")
        return [by_id[hid] for hid in wanted], len(wanted)

    top_n = os.environ.get("VP_ANALYZE_TOP_N", "all").strip().lower() or "all"
    if top_n == "all":
        # Test every accepted hypothesis, as the prototype did; none is left to fall back on.
        n = max(len(accepted), 1)
        default_attempts = n
    else:
        n = int(top_n)
        if n < 1:
            raise ValueError("VP_ANALYZE_TOP_N must be at least 1, or 'all'")
        default_attempts = n + 1
    attempts = int(os.environ.get("VP_ANALYZE_MAX_ATTEMPTS", str(default_attempts)))
    if attempts < n:
        raise ValueError("VP_ANALYZE_MAX_ATTEMPTS must be at least VP_ANALYZE_TOP_N")
    ranked = [h for _, h in sorted(enumerate(accepted),
                                   key=lambda p: _rank_key(reviews[p[1].id], p[0]))]
    ranking = "; ".join(
        f"{h.id} F{reviews[h.id].feasibility} N{reviews[h.id].novelty} "
        f"P{reviews[h.id].plausibility} A{reviews[h.id].alignment}"
        for h in ranked)
    queue = ranked[:attempts]
    bb.log("analyze", "selection",
           f"chose {', '.join(h.id for h in queue[:n])} of {len(ranked)} accepted"
           + (f", then {', '.join(h.id for h in queue[n:])} if one fails" if queue[n:] else "")
           + f"; ranked by feasibility, novelty, total score: {ranking}")
    return queue, n


async def run_analyze(bb: Blackboard, client: LLMClient, *, workdir: Path) -> None:
    """Top-level analyze pipeline. Async because the agentic analyst uses
    the Claude Agent SDK, which is async."""
    targets, wanted = _analysis_queue(bb)
    if not targets:
        message = ("No accepted hypotheses in the final iteration. Review the "
                   "hypothesis critiques; analysis and manuscript generation are stopped.")
        bb.log("analyze", "blocked", message)
        raise RuntimeError(message)

    max_iters = int(os.environ.get("VP_ANALYZE_MAX_METHODS_ITERS", "3"))
    if max_iters < 1:
        raise ValueError("VP_ANALYZE_MAX_METHODS_ITERS must be at least 1")

    print(f"\n--- agentic analyst (aiming to pass {wanted}, trying up to "
          f"{len(targets)} hypothesis/es) ---")
    analyst_model = client.resolve_model("sonnet")
    # Claude Code's background calls need a model this account can invoke
    # when it talks to Bedrock or the Anthropic API directly.
    background_model = ("" if getattr(client, "backend", "governor") == "governor"
                        else client.resolve_model("bulk"))
    passed: list[str] = []
    not_testable: dict[str, str] = {}
    unresolved: dict[str, str] = {}
    refused: dict[str, str] = {}
    for position, hyp in enumerate(targets):
        if len(passed) >= wanted:
            break
        if position >= wanted:
            # A hypothesis before this one did not pass, so this one takes its place.
            bb.log("analyze", "fallback",
                   f"{hyp.id}: trying the next-ranked hypothesis "
                   f"({len(passed)} of {wanted} passed so far)")
        for iteration in range(1, max_iters + 1):
            bb.log("analyze", "iteration", f"{hyp.id}: {iteration}/{max_iters}")
            folder = workdir / "hypotheses" / hyp.id
            try:
                # Sticky: after a refusal the rest of the stage uses the fallback.
                analyst_model = await _run_analyst(bb, hyp, client, workdir=workdir,
                                                   model_id=analyst_model,
                                                   background_model=background_model)
            except agentic_analyst.NotTestable as exc:
                _record_outcome(bb, hyp, folder, "not_testable",
                                f"the analyst found: {exc}", not_testable)
                break
            except AllAnalystsRefused as exc:
                _record_outcome(bb, hyp, folder, "refused", str(exc), refused)
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
    if not passed and refused:
        raise AllAnalystsRefused("; ".join(f"{h}: {r}" for h, r in refused.items()))


class AllAnalystsRefused(RuntimeError):
    """Every analyst model refused the task. The hypothesis is recorded as
    refused and the next-ranked one is tried; with none passed, the stage fails."""


async def _run_analyst(bb: Blackboard, hyp, client, *, workdir: Path, model_id: str,
                       background_model: str) -> str:
    """One analyst session, moved to the next fallback model each time the
    model refuses (`VP_ANALYZE_FALLBACK_MODEL`). Returns the model that ran it."""
    tried: list[str] = []
    usage_log = getattr(client, "usage_log", None)
    while True:
        try:
            await agentic_analyst.run(bb, hyp, workdir=workdir, model_id=model_id,
                                      background_model=background_model, usage_log=usage_log,
                                      can_invoke=getattr(client, "can_invoke", None))
            return model_id
        except agentic_analyst.AnalystRefused:
            tried.append(model_id)
            fallback = next((m for m in agentic_analyst.analyst_fallbacks(
                                 model_id, getattr(client, "can_invoke", None))
                             if m not in tried), None)
            if usage_log is not None:
                usage_log.append({"event": "refusal", "agent": "agentic_analyst",
                                  "hypothesis_id": hyp.id, "model_id": model_id,
                                  "fallback_model_id": fallback})
            if fallback is None:
                raise AllAnalystsRefused(
                    f"every analyst model refused the task ({', '.join(tried)}); "
                    f"set VP_ANALYZE_FALLBACK_MODEL to try another") from None
            bb.log("analyze", "refusal_fallback",
                   f"{hyp.id}: {model_id} refused; rerunning the round on {fallback}")
            model_id = fallback


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
