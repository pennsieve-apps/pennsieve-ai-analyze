"""Agentic Analyst — Claude-Code-driven notebook generation + execution.

For one hypothesis on the blackboard, kicks off a Claude Code agent
session with Read/Write/Edit/Bash tools that:
  1. Inspects the dataset(s) named in the hypothesis (real columns,
     dtypes, NaN rates — not blind guesses).
  2. Drafts a Jupyter notebook that tests the claim.
  3. Executes it via papermill.
  4. Fixes errors in-loop.
  5. Returns only after a successful SDK result and a cleanly executed
     notebook; turn-budget exhaustion is a stage failure.

Auth model: the Agent SDK starts the Claude Code CLI inside this container.
It never uses anyone's Claude subscription or login; it authenticates only
from environment variables. On the platform (`VP_LLM_BACKEND=governor`) it
reads `ANTHROPIC_BASE_URL`, set by the platform to `http://localhost:8765`, the
in-task sidecar that bridges to the LLM governor via
`lambda:InvokeWithResponseStream`. For local runs, `VP_LLM_BACKEND=bedrock`
switches it to Bedrock with the caller's AWS credentials, and `anthropic` to
the Anthropic API with `ANTHROPIC_API_KEY` (see `pennsieve_ai_utils.backend`).

Scaffold-stage notes:
  - The full prototype in `virtual-postdoc` carries SPARC-specific dataset
    inspection, multi-dataset role decomposition, and an AUP-refusal
    fallback chain. Those are deliberately omitted here so the scaffold
    is auditable; port them piecewise as the use case demands.
  - Cost is tracked via the SDK's `ResultMessage.total_cost_usd` and
    appended to the run's usage.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from pennsieve_ai_utils.backend import claude_code_env
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis
from pennsieve_ai_utils.llm import raw_io_enabled

# Maximum agent turns. Generous so the agent has room to iterate on
# real-world data quirks; the agent should self-terminate well before.
MAX_TURNS = int(os.environ.get("VP_ANALYZE_MAX_TURNS", "80"))

# Tool results are cut to this many characters in the trace, unless
# VP_LOG_LLM_IO=1 asks for the full transcript.
TRACE_RESULT_CHARS = 8000


async def run(
    bb: Blackboard,
    hypothesis: Hypothesis,
    *,
    workdir: Path,
    model_id: str,
    background_model: str = "",
) -> None:
    """Drive one analyst session against one hypothesis.

    `background_model` serves Claude Code's own small background calls on a
    direct backend; the governor path leaves that to the platform.
    """
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ClaudeSDKClient,
            PermissionResultAllow,
            PermissionResultDeny,
        )
    except ImportError as e:
        bb.log(
            "agentic_analyst", "missing_dependency",
            "claude-agent-sdk is required for the analyze stage; install "
            "with `pip install claude-agent-sdk`. Original error: " + repr(e),
        )
        raise RuntimeError("claude-agent-sdk is required for the analyze stage") from e

    # Per-hypothesis working subdirectory. The Claude Code session runs
    # with this as cwd so any files the agent writes are scoped here.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", hypothesis.id):
        raise ValueError("Hypothesis ID must be a safe directory name")
    hyp_workdir = (workdir / "hypotheses" / hypothesis.id).resolve()
    hyp_workdir.mkdir(parents=True, exist_ok=True)

    task_prompt = _build_task_prompt(bb, hypothesis, hyp_workdir)
    print(
        f"  [agent] hypothesis {hypothesis.id}: model={model_id} cwd={hyp_workdir}",
        flush=True,
    )

    # Approve the analysis tool set without interactive prompts. Isolation
    # is supplied by the compute runtime; cwd itself is not a sandbox.
    async def _allow_all(tool_name, tool_input, context):
        if tool_name in {"Read", "Write", "Edit", "Bash", "NotebookEdit"}:
            return PermissionResultAllow()
        return PermissionResultDeny(message="Tool is not enabled for this analysis")

    options = ClaudeAgentOptions(
        model=model_id,
        max_turns=MAX_TURNS,
        # Broad allowed_tools entries bypass can_use_tool in the SDK.
        # Leave approval to the callback; cwd is not a security boundary.
        can_use_tool=_allow_all,
        cwd=str(hyp_workdir),
        env=claude_code_env(background_model),
    )

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tools_called: list[str] = []
    total_cost = 0.0
    final_text = ""
    completed = False
    trace_path = hyp_workdir / "agent-events.jsonl"

    def event(kind, **fields):
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "hypothesis_id": hypothesis.id, "event": kind, **fields}
        text = json.dumps(record, default=str)
        # Remove environment credentials and common inline secrets from traces.
        for key, value in os.environ.items():
            if any(word in key.upper() for word in ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "ACCESS_KEY")) and len(value) > 5:
                text = text.replace(value, "[REDACTED]")
        text = re.sub(r"(?i)(X-Amz-[^=\s&]+)=([^&\s\"]+)", r"\1=[REDACTED]", text)
        with trace_path.open("a") as trace:
            trace.write(text + "\n")
        return json.loads(text)

    # With VP_LOG_LLM_IO=1 the trace is a full transcript: the task prompt,
    # every text and thinking block, untruncated tool results, and per-turn
    # usage. Claude Code's own system prompt is not exposed by the SDK.
    raw = raw_io_enabled()
    event("start", model=model_id, max_turns=MAX_TURNS,
          **({"prompt": task_prompt} if raw else {}))

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(task_prompt)
            async for msg in client.receive_response():
                content = getattr(msg, "content", None) or []
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "name") and hasattr(block, "input"):
                            tools_called.append(block.name)
                            safe = event("tool_start", tool=block.name,
                                         tool_use_id=getattr(block, "id", None), input=block.input)
                            arguments = safe["input"]
                            detail = (arguments.get("description") or arguments.get("command") or
                                      arguments.get("file_path") or arguments.get("notebook_path") or "")
                            print(f"  [agent] {hypothesis.id} tool #{len(tools_called)}: "
                                  f"{block.name} id={getattr(block, 'id', '?')} "
                                  f"{str(detail).replace(chr(10), ' ')[:160]}", flush=True)
                        elif hasattr(block, "tool_use_id"):
                            result = str(getattr(block, "content", ""))
                            event("tool_result", tool_use_id=block.tool_use_id,
                                  is_error=getattr(block, "is_error", False),
                                  content=result if raw else result[:TRACE_RESULT_CHARS])
                        elif hasattr(block, "thinking"):
                            if raw:
                                event("thinking", text=block.thinking)
                        elif hasattr(block, "text"):
                            final_text = block.text
                            if raw:
                                event("text", text=block.text)
                usage = getattr(msg, "usage", None)
                if raw and usage and not hasattr(msg, "is_error"):
                    event("turn_usage", model=getattr(msg, "model", None), usage=usage)
                cost = getattr(msg, "total_cost_usd", None)
                if cost is not None:
                    total_cost = max(total_cost, float(cost))
                if hasattr(msg, "is_error"):
                    event("result", subtype=getattr(msg, "subtype", ""),
                          is_error=msg.is_error, cost_usd=total_cost,
                          usage=getattr(msg, "usage", {}))
                    if msg.is_error or getattr(msg, "subtype", "") != "success":
                        raise RuntimeError(f"Agent did not complete: {getattr(msg, 'subtype', '?')}")
                    completed = True
                    final_text = getattr(msg, "result", None) or final_text
        if not completed or not final_text.strip():
            raise RuntimeError("Agent ended without a successful result and summary")
        validate_notebook(hyp_workdir / "analysis.ipynb")
    except Exception as e:
        hypothesis.analysis_cost_usd += total_cost
        hypothesis.analysis_tools_called.extend(tools_called)
        bb.log(
            "agentic_analyst", "error",
            f"hypothesis {hypothesis.id}: agent loop failed: {e!r}",
        )
        raise RuntimeError(
            f"Agentic analysis failed for hypothesis {hypothesis.id}"
        ) from e

    # Record artifacts on the hypothesis. The Write stage reads these to
    # produce the final report.
    hypothesis.analysis_workdir = str(hyp_workdir.relative_to(workdir.resolve().parent))
    hypothesis.analysis_summary = final_text[:8000]  # cap to keep blackboard.json sane
    hypothesis.analysis_cost_usd += total_cost
    hypothesis.analysis_tools_called.extend(tools_called)
    hypothesis.analysis_started_at = started
    hypothesis.analysis_finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    bb.log(
        "agentic_analyst", "done",
        f"hypothesis {hypothesis.id}: tools={len(tools_called)} "
        f"cost=${total_cost:.4f}",
    )


def _build_task_prompt(bb: Blackboard, hypothesis: Hypothesis, workdir: Path) -> str:
    """Render the initial task prompt the analyst receives. Kept minimal in
    the scaffold; the prototype version is far richer (per-dataset blocks,
    role sequences for cross-dataset hypotheses, narrowed-claim handling).
    """
    venv_python = sys.executable
    datasets_block = _format_datasets_brief(bb)
    revision = ("\n## Required methods revisions\nRevise the existing notebook and re-execute ALL cells.\n"
                + json.dumps(hypothesis.methods_critique, indent=2)
                if hypothesis.methods_critique else "")
    return f"""You are running a single analysis to test one hypothesis.

You have Read, Write, Edit, and Bash tools. Use Bash to run Python:
`{venv_python}` has pandas, scipy, statsmodels, matplotlib, papermill,
nbformat, jupytext, httpx, and openpyxl pre-installed.

If you need a package that's not pre-installed (e.g., `lifelines`,
`pynwb`, `nibabel`), install it with `pip install --user <package>`.
The `--user` flag is required because this container runs without
root privileges; standard `pip install` will fail.

## Hypothesis {hypothesis.id}

**Claim to test:** {hypothesis.claim}

**Rationale:** {hypothesis.rationale}
{revision}

## Datasets available

{datasets_block}

## Your task

1. Inspect the dataset(s) named in the hypothesis. Look at actual columns,
   dtypes, NaN rates — do NOT assume schema from the title alone.
2. Write a Jupyter notebook (`analysis.ipynb`) that tests the claim
   end-to-end: load → preprocess → analysis → conclusion.
3. Run it via papermill. Fix errors iteratively.
   Save the executed output as `analysis.ipynb`; every nonempty code cell
   must have an execution count. Never substitute synthetic data for the
   deposit. If the data cannot support the test, report the blocker.
4. When the notebook runs cleanly, write a short summary of what you
   found and whether the claim is supported, partially supported, or
   not supported by the data. Be honest about assumptions and limits.

Your working directory is `{workdir}`. Write all files there.
"""


def _format_datasets_brief(bb: Blackboard) -> str:
    if not bb.dataset_summaries:
        return "(no datasets on the blackboard — Hypothesize did not record any)"
    out = []
    for d in bb.dataset_summaries:
        out.append(
            f"- **{d.id}**: {d.title}\n"
            f"  - Bucket: {d.bucket_uri or '(none)'}\n"
            f"  - Access: {d.access_type or 'unknown'}\n"
            f"  - {d.description[:300]}"
            f"\n  - Repository URL: {d.url}"
            f"\n  - README: {d.readme[:1500]}"
            f"\n  - File tree: {d.file_tree[:10000]}"
            f"\n  - Record tables: {json.dumps(d.record_tables)[:3000]}"
        )
    return "\n".join(out)


def validate_notebook(path: Path) -> None:
    """Require actual executed cells; absence of errors alone is insufficient."""
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    if not code or any(c.execution_count is None for c in code):
        raise RuntimeError("analysis.ipynb has no executed code or contains unexecuted cells")
    if any(o.output_type == "error" for c in code for o in c.outputs):
        raise RuntimeError("analysis.ipynb contains execution errors")
