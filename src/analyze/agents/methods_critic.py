"""Methods Critic — single-shot LLM review of the agentic analyst's output.

Not an agent loop: a plain Opus call that reads the analyst's summary
text + the contents of the produced notebook and flags methodological
issues (untested assumptions, inappropriate stats, missing controls,
weak conclusions). The output is attached to the hypothesis so the
Write stage can cite or rebut.

Kept deliberately simple in the scaffold — the prototype's full critic
runs a structured prompt with rubric-style sections; that can land in
a follow-up. For v0 we just produce a free-text critique.
"""
from __future__ import annotations

import os
from pathlib import Path

from analyze.agents._common import call_json
from analyze.blackboard import Blackboard, Hypothesis
from analyze.llm import LLMClient


def run(bb: Blackboard, client: LLMClient, hypothesis: Hypothesis) -> None:
    if not hypothesis.analysis_summary:
        bb.log("methods_critic", "skipped",
               f"hypothesis {hypothesis.id}: no analyst summary to critique")
        return

    notebook_text = _read_notebook_text(hypothesis)

    system = (
        "You are a methods critic reviewing an automated data analysis. "
        "Be honest, specific, and concise. Flag failed assumptions, "
        "inappropriate statistical choices, missing controls, and weak "
        "inferences. If the analysis is sound, say so plainly — no false "
        "positives."
    )
    user = f"""## Hypothesis
{hypothesis.claim}

## Rationale
{hypothesis.rationale}

## Analyst summary
{hypothesis.analysis_summary}

## Notebook (truncated)
```
{notebook_text[:20000]}
```

Produce a JSON object with this shape:

{{
  "verdict": "sound" | "concerns" | "unsound",
  "issues": [
    {{"severity": "high|medium|low", "section": "<notebook section or 'overall'>", "description": "..."}}
  ],
  "suggested_followups": ["..."]
}}
"""

    try:
        result = call_json(
            client, agent="methods_critic", system=system, user=user,
            run_id=bb.run_id, model="opus", max_tokens=4096,
        )
    except Exception as e:
        bb.log("methods_critic", "error",
               f"hypothesis {hypothesis.id}: critique call failed: {e!r}")
        return

    hypothesis.methods_critique = result
    bb.log("methods_critic", "done",
           f"hypothesis {hypothesis.id}: verdict={result.get('verdict', '?')}, "
           f"issues={len(result.get('issues', []))}")


def _read_notebook_text(hypothesis: Hypothesis) -> str:
    """Best-effort read of the produced notebook. Returns "" if it doesn't
    exist or can't be read — the critic can still produce a useful review
    from the analyst summary alone."""
    workdir_rel = hypothesis.analysis_workdir
    if not workdir_rel:
        return ""
    # analysis_workdir was stored as relative path from output_dir.parent
    # so we re-anchor to whatever output_dir is now. In a workflow run
    # those are the same path; in a local re-run they may differ.
    output_dir = Path(os.environ.get("OUTPUT_DIR", ""))
    if output_dir:
        candidate = output_dir.parent / workdir_rel / "analysis.ipynb"
        if candidate.exists():
            try:
                return candidate.read_text()
            except Exception:
                pass
    return ""
