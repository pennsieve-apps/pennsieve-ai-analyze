"""Methods Critic — single-shot LLM review of the agentic analyst's output.

Not an agent loop: a direct reasoning-role call that reads the analyst's
summary text + the contents of the produced notebook and flags methodological
issues (untested assumptions, inappropriate stats, missing controls, weak
conclusions). The output is attached to the hypothesis so the Write stage can
cite or rebut.

Kept deliberately simple in the scaffold — the prototype's full critic
runs a structured prompt with rubric-style sections; that can land in
a follow-up. For v0 we just produce a free-text critique.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from analyze.agents._common import call_json
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis
from pennsieve_ai_utils.llm import LLMClient


def run(bb: Blackboard, client: LLMClient, hypothesis: Hypothesis, *, notebook_path: Path | None = None) -> None:
    if not hypothesis.analysis_summary:
        bb.log("methods_critic", "skipped",
               f"hypothesis {hypothesis.id}: no analyst summary to critique")
        raise RuntimeError("Methods critic requires an analyst summary")

    notebook_text = notebook_path.read_text() if notebook_path else _read_notebook_text(hypothesis)
    if not notebook_text:
        raise RuntimeError("Methods critic requires the executed notebook, not just a summary")
    notebook_text = _compact_notebook(notebook_text)
    status_path = (notebook_path.parent if notebook_path else Path(".")) / "analysis_status.json"
    status_text = status_path.read_text() if status_path.is_file() else "(the analyst wrote no status file)"

    system = (
        "You are a methods critic reviewing an automated data analysis. "
        "Be honest, specific, and concise. Flag failed assumptions, "
        "inappropriate statistical choices, missing controls, and weak "
        "inferences. If the analysis is sound, say so plainly — no false "
        "positives. An analysis that did not read and analyze the deposit's "
        "own data (for example, a notebook that only reports that the data "
        "could not be downloaded, or that uses synthetic or made-up values) "
        "is never sound: give the verdict \"unsound\" with a high-severity "
        "issue saying what was not analyzed."
    )
    user = f"""## Hypothesis
{hypothesis.claim}

## Rationale
{hypothesis.rationale}

## Analyst summary
{hypothesis.analysis_summary}

## Analyst's status report (analysis_status.json)
{status_text}

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
        raise RuntimeError(
            f"Methods critique failed for hypothesis {hypothesis.id}"
        ) from e

    if result.get("verdict") not in {"sound", "concerns", "unsound"} or not isinstance(result.get("issues"), list):
        raise ValueError("Methods critic returned an invalid verdict or issues list")
    if any(not isinstance(i, dict) or i.get("severity") not in {"high", "medium", "low"}
           for i in result["issues"]):
        raise ValueError("Methods critic returned an invalid issue")
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


def _compact_notebook(raw: str) -> str:
    """Show sources and outputs instead of spending the context on base64 PNGs."""
    notebook = json.loads(raw)
    cells = notebook.get("cells", [])
    per_cell = max(200, 19000 // max(len(cells), 1))
    blocks = []
    for index, cell in enumerate(cells):
        outputs = []
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain", "")
            if text:
                outputs.append("".join(text))
            if output.get("output_type") == "error":
                outputs.append(f"ERROR: {output.get('ename')}: {output.get('evalue')}")
        source = "".join(cell.get("source", ""))
        blocks.append(f"Cell {index} ({cell.get('cell_type')}, execution={cell.get('execution_count')}):\n"
                      f"{source[:per_cell // 2]}\nOutputs:\n{chr(10).join(outputs)[:per_cell // 2]}")
    return "\n\n".join(blocks)
