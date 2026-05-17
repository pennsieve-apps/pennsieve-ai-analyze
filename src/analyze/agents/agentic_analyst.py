"""Agentic Analyst — Claude-Code-driven notebook generation + execution.

For one hypothesis on the blackboard, kicks off a Claude Code agent
session with Read/Write/Edit/Bash tools that:
  1. Inspects the dataset(s) named in the hypothesis (real columns,
     dtypes, NaN rates — not blind guesses).
  2. Drafts a Jupyter notebook that tests the claim.
  3. Executes it via papermill.
  4. Fixes errors in-loop.
  5. Returns when the notebook runs cleanly OR the turn budget is hit.

Auth model: the Claude Agent SDK reads `ANTHROPIC_BASE_URL` from env
(set by the platform to `http://localhost:8765` — the in-task sidecar).
The sidecar bridges to the LLM governor via `lambda:InvokeWithResponseStream`.
We don't configure auth here.

Scaffold-stage notes:
  - The full prototype in `virtual-postdoc` carries SPARC-specific dataset
    inspection, multi-dataset role decomposition, and an AUP-refusal
    fallback chain. Those are deliberately omitted here so the scaffold
    is auditable; port them piecewise as the use case demands.
  - Cost is tracked via the SDK's `ResultMessage.total_cost_usd` and
    appended to the run's usage.jsonl.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from analyze.blackboard import Blackboard, Hypothesis

# Maximum agent turns. Generous so the agent has room to iterate on
# real-world data quirks; the agent should self-terminate well before.
MAX_TURNS = int(os.environ.get("VP_ANALYZE_MAX_TURNS", "80"))

# Model selection. The governor's allow-list is the source of truth for
# what's actually permitted; this list is just the preferred order. Each
# entry is a Bedrock inference-profile ID (us.* for HIPAA-friendly US
# region routing).
DEFAULT_MODEL = os.environ.get(
    "VP_ANALYZE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)

# How much of a tool call's input/output to surface in the log. Big enough
# to be useful (full bash commands, file paths, first line of output), small
# enough that CloudWatch doesn't get flooded when the agent is busy.
_TOOL_INPUT_PREVIEW_CHARS = 200
_TOOL_RESULT_PREVIEW_CHARS = 200


def _truncate(s: str, limit: int) -> str:
    """Truncate `s` to `limit` chars, appending an ellipsis if cut."""
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _format_tool_input(name: str, tool_input: dict) -> str:
    """Render a tool_use block's input as a one-line CloudWatch preview.

    Each tool gets a hand-picked shape so the log is actually readable:
    Bash → the command, Read/Write/Edit → the path, Grep → pattern + path.
    Unknown tools fall back to a truncated repr of the input dict.
    """
    if name == "Bash":
        cmd = tool_input.get("command", "")
        return _truncate(f"$ {cmd}", _TOOL_INPUT_PREVIEW_CHARS)
    if name == "Read":
        return tool_input.get("file_path", "")
    if name == "Write":
        path = tool_input.get("file_path", "")
        body = tool_input.get("content", "") or ""
        return f"{path} ({len(body)}B)"
    if name == "Edit":
        path = tool_input.get("file_path", "")
        old = (tool_input.get("old_string", "") or "").split("\n", 1)[0]
        return f"{path} | {_truncate(old, 80)} → …"
    if name == "Grep":
        return (f"{tool_input.get('pattern', '')} "
                f"in {tool_input.get('path', '.')}")
    if name == "Glob":
        return tool_input.get("pattern", "")
    return _truncate(str(tool_input), 120)


def _format_tool_result(content) -> str:
    """Render a tool_result block's content as a one-line CloudWatch preview.

    SDK delivers content as a string (legacy) or a list of typed blocks
    ({"type": "text", "text": "..."} or similar). We pull out the first
    line and cap it. Anything we can't parse falls back to a short repr.
    """
    if isinstance(content, str):
        return _truncate(content.split("\n", 1)[0], _TOOL_RESULT_PREVIEW_CHARS)
    if isinstance(content, list):
        for block in content:
            text = None
            if isinstance(block, dict):
                text = block.get("text")
            elif hasattr(block, "text"):
                text = block.text
            if text:
                return _truncate(text.split("\n", 1)[0],
                                 _TOOL_RESULT_PREVIEW_CHARS)
    return _truncate(str(content), 120)


async def run(bb: Blackboard, hypothesis: Hypothesis, *, workdir: Path) -> None:
    """Drive one analyst session against one hypothesis."""
    try:
        from claude_agent_sdk import (
            ClaudeSDKClient, ClaudeAgentOptions, PermissionResultAllow,
        )
    except ImportError as e:
        bb.log(
            "agentic_analyst", "missing_dependency",
            "claude-agent-sdk is required for the analyze stage; install "
            "with `pip install claude-agent-sdk`. Original error: " + repr(e),
        )
        return

    # Per-hypothesis working subdirectory. The Claude Code session runs
    # with this as cwd so any files the agent writes are scoped here.
    hyp_workdir = workdir / "hypotheses" / hypothesis.id
    hyp_workdir.mkdir(parents=True, exist_ok=True)

    task_prompt = _build_task_prompt(bb, hypothesis, hyp_workdir)
    print(f"  [agent] hypothesis {hypothesis.id}: cwd={hyp_workdir}", flush=True)

    # Auto-approve every tool call via the can_use_tool callback. Safety
    # story is the surrounding sandbox: the container runs as uid 1000
    # (platform-enforced), the per-execution EFS access point caps
    # filesystem reach to /data/{cn}/{run}/, the governor enforces the
    # LLM budget, and the task IAM role is execution-scoped. The CLI's
    # interactive prompt would just block in a non-interactive container
    # anyway — `can_use_tool` makes the auto-approve explicit instead.
    async def _allow_all(tool_name, tool_input, context):
        return PermissionResultAllow()

    options = ClaudeAgentOptions(
        model=DEFAULT_MODEL,
        max_turns=MAX_TURNS,
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        can_use_tool=_allow_all,
        cwd=str(hyp_workdir),
    )

    started = datetime.utcnow().isoformat(timespec="seconds")
    tools_called: list[str] = []
    total_cost = 0.0
    final_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(task_prompt)
            async for msg in client.receive_response():
                content = getattr(msg, "content", None) or []
                if isinstance(content, list):
                    for block in content:
                        # tool_use: agent invoking a tool. Log the tool name
                        # plus a short preview of the input (Bash command,
                        # file path, etc.) so CloudWatch shows what the
                        # agent is actually doing, not just that it's busy.
                        if hasattr(block, "name") and hasattr(block, "input"):
                            tools_called.append(block.name)
                            preview = _format_tool_input(
                                block.name, block.input or {})
                            print(f"  [agent] {block.name}: {preview}",
                                  flush=True)
                        # tool_result: response coming back from the tool.
                        # First line, capped at 200 chars. ✗ marks errors.
                        elif (hasattr(block, "tool_use_id")
                              and hasattr(block, "content")):
                            marker = ("✗"
                                      if getattr(block, "is_error", False)
                                      else "→")
                            result = _format_tool_result(block.content)
                            print(f"  [agent] {marker} {result}", flush=True)
                        # text: the agent's reasoning between tool calls,
                        # and (at the end) its final summary. We print
                        # every text block and track the last one as the
                        # summary written to the blackboard.
                        elif hasattr(block, "text"):
                            text = (block.text or "").strip()
                            if text:
                                for line in text.split("\n"):
                                    print(f"  [agent] {line}", flush=True)
                                final_text = block.text
                cost = getattr(msg, "total_cost_usd", None)
                if cost is not None:
                    total_cost = max(total_cost, float(cost))
    except Exception as e:
        bb.log(
            "agentic_analyst", "error",
            f"hypothesis {hypothesis.id}: agent loop failed: {e!r}",
        )
        return

    # Record artifacts on the hypothesis. The Write stage reads these to
    # produce the final report.
    hypothesis.analysis_workdir = str(hyp_workdir.relative_to(workdir.parent))
    hypothesis.analysis_summary = final_text[:8000]  # cap to keep blackboard.json sane
    hypothesis.analysis_cost_usd = total_cost
    hypothesis.analysis_tools_called = tools_called
    hypothesis.analysis_started_at = started
    hypothesis.analysis_finished_at = datetime.utcnow().isoformat(timespec="seconds")

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

## Datasets available

{datasets_block}

## Your task

1. Inspect the dataset(s) named in the hypothesis. Look at actual columns,
   dtypes, NaN rates — do NOT assume schema from the title alone.
2. Write a Jupyter notebook (`analysis.ipynb`) that tests the claim
   end-to-end: load → preprocess → analysis → conclusion.
3. Run it via papermill. Fix errors iteratively.
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
        )
    return "\n".join(out)
