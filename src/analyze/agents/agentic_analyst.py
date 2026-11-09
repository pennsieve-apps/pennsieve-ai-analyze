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

from pennsieve_ai_utils.backend import claude_code_env, llm_backend
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis
from pennsieve_ai_utils.llm import raw_io_enabled
from pennsieve_ai_utils.models import OPUS_FALLBACKS

from analyze import data_staging

# Maximum agent turns. Generous so the agent has room to iterate on
# real-world data quirks; the agent should self-terminate well before.
MAX_TURNS = int(os.environ.get("VP_ANALYZE_MAX_TURNS", "80"))

# The pennsieve-mcp server, when the run provides one (VP_PENNSIEVE_MCP_URL).
# Claude Code names its tools mcp__<server>__<tool>.
MCP_SERVER = "pennsieve"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER}__"

# The analyst's models when its first model refuses the task: the shared
# Opus chain, newest first. Opus 5.5 refused seed 405; the older Opus models
# have not refused any of the pipeline's datasets.
DEFAULT_ANALYST_FALLBACKS = OPUS_FALLBACKS


def analyst_fallbacks(model_id: str, can_invoke=None) -> list[str]:
    """Models to try, in order, after `model_id` refuses the task.

    `VP_ANALYZE_FALLBACK_MODEL` is a comma-separated list that replaces the
    default; `none` turns the fallback off. With `can_invoke` (usually
    `LLMClient.can_invoke`), models the account cannot call are dropped.
    """
    configured = os.environ.get("VP_ANALYZE_FALLBACK_MODEL", "").strip()
    if configured.lower() in ("none", "off"):
        return []
    models = [m.strip() for m in configured.split(",") if m.strip()] or list(DEFAULT_ANALYST_FALLBACKS)
    return [m for m in models if m != model_id and (can_invoke is None or can_invoke(m))]


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
    can_invoke=None,
    usage_log: list[dict] | None = None,
) -> None:
    """Drive one analyst session against one hypothesis.

    `background_model` serves Claude Code's own small background calls on a
    direct backend; the governor path leaves that to the platform.
    `usage_log` is the stage's usage list (`LLMClient.usage_log`); the
    session's token counts are added to it as one `agent_sdk_result` row.
    `analyst_fallbacks` also go to Claude Code's own fallback, which covers a
    model that is overloaded or unavailable. A refusal raises `AnalystRefused`.
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

    # Download the dataset's file index and a sample before the agent starts,
    # through the public Discover route. Raises DataStagingError if the
    # dataset cannot be reached, so the stage stops instead of guessing.
    staging = data_staging.stage(bb, hypothesis, hyp_workdir)
    mcp_url = os.environ.get("VP_PENNSIEVE_MCP_URL", "").strip()
    task_prompt = _build_task_prompt(bb, hypothesis, hyp_workdir, staging)
    print(
        f"  [agent] hypothesis {hypothesis.id}: model={model_id} cwd={hyp_workdir}",
        flush=True,
    )

    # Approve the analysis tool set without interactive prompts. Isolation
    # is supplied by the compute runtime; cwd itself is not a sandbox.
    async def _allow_all(tool_name, tool_input, context):
        if tool_name in {"Read", "Write", "Edit", "Bash", "NotebookEdit"}:
            return PermissionResultAllow()
        if mcp_url and tool_name.startswith(MCP_TOOL_PREFIX):
            return PermissionResultAllow()
        return PermissionResultDeny(message="Tool is not enabled for this analysis")

    options = ClaudeAgentOptions(
        # The pennsieve-mcp server's public Discover tools, when the run provides one.
        mcp_servers={MCP_SERVER: {"type": "http", "url": mcp_url}} if mcp_url else {},
        model=model_id,
        max_turns=MAX_TURNS,
        # Broad allowed_tools entries bypass can_use_tool in the SDK.
        # Leave approval to the callback; cwd is not a security boundary.
        can_use_tool=_allow_all,
        cwd=str(hyp_workdir),
        env=claude_code_env(background_model),
        # Claude Code takes one fallback, for a model that is overloaded or unavailable.
        fallback_model=next(iter(analyst_fallbacks(model_id, can_invoke)), None),
    )

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tools_called: list[str] = []
    total_cost = 0.0
    final_text = ""
    completed = False
    refused = False
    # The SDK reports the session's usage only on its result message, so a
    # session that is cancelled or crashes spends real money and leaves no
    # usage row. Keep a running total from the per-turn usage instead, and
    # record it if the result never arrives.
    turn_totals = dict.fromkeys(USAGE_KEYS, 0)
    turns_seen = 0
    usage_recorded = False
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
                if getattr(msg, "stop_reason", None) == "refusal" and not refused:
                    refused = True
                    event("refusal", model=getattr(msg, "model", None) or model_id)
                usage = getattr(msg, "usage", None)
                if usage and not hasattr(msg, "is_error"):
                    turns_seen += 1
                    for key in USAGE_KEYS:
                        turn_totals[key] += _count(usage, key)
                    if raw:
                        event("turn_usage", model=getattr(msg, "model", None), usage=usage)
                cost = getattr(msg, "total_cost_usd", None)
                if cost is not None:
                    total_cost = max(total_cost, float(cost))
                if hasattr(msg, "is_error"):
                    event("result", subtype=getattr(msg, "subtype", ""),
                          is_error=msg.is_error, cost_usd=total_cost,
                          usage=getattr(msg, "usage", {}))
                    _record_sdk_usage(usage_log, hypothesis.id, model_id,
                                      getattr(msg, "usage", None) or {}, total_cost,
                                      len(tools_called), getattr(msg, "num_turns", None))
                    usage_recorded = True
                    if refused:
                        raise AnalystRefused(model_id)
                    if msg.is_error or getattr(msg, "subtype", "") != "success":
                        raise RuntimeError(f"Agent did not complete: {getattr(msg, 'subtype', '?')}")
                    completed = True
                    final_text = getattr(msg, "result", None) or final_text
        if refused:
            raise AnalystRefused(model_id)
        if not completed or not final_text.strip():
            raise RuntimeError("Agent ended without a successful result and summary")
        validate_notebook(hyp_workdir / "analysis.ipynb")
        check_analysis_plan(hyp_workdir)
        check_analysis_status(hyp_workdir)
    except NotTestable as e:
        hypothesis.analysis_cost_usd += total_cost
        hypothesis.analysis_tools_called.extend(tools_called)
        hypothesis.analysis_workdir = str(hyp_workdir.relative_to(workdir.resolve().parent))
        hypothesis.analysis_summary = final_text[:8000]
        bb.log("agentic_analyst", "not_testable", f"hypothesis {hypothesis.id}: {e}")
        raise
    except AnalystRefused as e:
        hypothesis.analysis_cost_usd += total_cost
        hypothesis.analysis_tools_called.extend(tools_called)
        bb.log("agentic_analyst", "refused", f"hypothesis {hypothesis.id}: {e}")
        raise
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
    finally:
        # Whatever ended the session — a cancelled timeout, a crash, a refusal —
        # the turns it took were billed. Record them so the run's cost is the
        # cost of the run.
        if not usage_recorded:
            _record_sdk_usage(usage_log, hypothesis.id, model_id, turn_totals, total_cost,
                              len(tools_called), turns_seen, partial=True)

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


USAGE_KEYS = ("input_tokens", "output_tokens",
              "cache_creation_input_tokens", "cache_read_input_tokens")


def _count(usage, key: str) -> int:
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
    return int(value or 0)


def _record_sdk_usage(usage_log, hypothesis_id, model_id, sdk_usage, sdk_cost, tools, turns,
                      *, partial: bool = False) -> None:
    """Add the analyst session's token counts to the stage usage log.

    The Agent SDK reports the session's total usage on its result message. A
    session that never reaches one still spent its turns, so `partial` marks a
    row summed from the per-turn usage instead.
    Only token counts are stored, under the same field names as the other
    usage rows, so cost can be worked out later from a price table. The SDK's
    own `total_cost_usd` comes from Claude Code's built-in Anthropic list
    prices, not Bedrock's; it is kept, labeled as the SDK's figure, so the two
    can be compared.
    """
    if usage_log is None:
        return

    def count(key):
        return _count(sdk_usage, key)

    usage_log.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "agent_sdk_result",
        "agent": "agentic_analyst",
        "hypothesis_id": hypothesis_id,
        "backend": llm_backend(),
        "model_id": model_id,
        "input_tokens": count("input_tokens"),
        "output_tokens": count("output_tokens"),
        "cache_creation_input_tokens": count("cache_creation_input_tokens"),
        "cache_read_input_tokens": count("cache_read_input_tokens"),
        "tools_called": tools,
        "turns": turns,
        "sdk_reported_cost_usd": float(sdk_cost),
        # Summed from the turns rather than read off a result message that
        # never came, so the session's tokens still reach the cost total.
        **({"partial": True} if partial else {}),
    })


def _build_task_prompt(bb: Blackboard, hypothesis: Hypothesis, workdir: Path,
                       staging: data_staging.StagingResult | None = None) -> str:
    """Render the initial task prompt the analyst receives. Kept minimal in
    the scaffold; the prototype version is far richer (per-dataset blocks,
    role sequences for cross-dataset hypotheses, narrowed-claim handling).
    """
    venv_python = sys.executable
    datasets_block = _format_datasets_brief(bb)
    revision = ("\n## Required methods revisions\nRevise the existing notebook and re-execute ALL cells.\n"
                + json.dumps(hypothesis.methods_critique, indent=2)
                if hypothesis.methods_critique else "")
    data_access = data_staging.prompt_section(staging) if staging else ""
    if os.environ.get("VP_PENNSIEVE_MCP_URL", "").strip():
        data_access += "\n\n" + PENNSIEVE_TOOLS_SECTION
    return f"""You are running a single analysis to test one hypothesis.

You have Read, Write, Edit, and Bash tools. Use Bash to run Python:
`{venv_python}` has pandas, scipy, statsmodels, matplotlib, papermill,
nbformat, jupytext, httpx, openpyxl and flowio pre-installed. Use flowio to
read FCS files. Its `text` dictionary includes the `$SPILL` matrix; apply it
to compensate the fluorescence channels before gating.

If you need a package that's not pre-installed (e.g., `lifelines`,
`pynwb`, `nibabel`), install it with `pip install --user <package>`.
Do not install anything that upgrades or downgrades numpy, pandas or scipy;
other cells and the methods review depend on the installed versions.
The `--user` flag is required because this container runs without
root privileges; standard `pip install` will fail.

## Hypothesis {hypothesis.id}

**Claim to test:** {hypothesis.claim}

**Rationale:** {hypothesis.rationale}

**Predicted observation (pre-specified):** {hypothesis.predicted_observation}

**Experiment to run (pre-specified):** {hypothesis.suggested_experiment}

The predicted observation and the experiment above are the approved plan: a
hypothesis critic accepted this hypothesis on the strength of that specific
test. They are what you run. Troubleshooting is expected — fixing code,
chasing down file layouts, correcting a gate, making a tool call work. Changing
*what is tested* is not: do not drop an arm the claim names, do not swap the
metric the prediction states, do not narrow the comparison, and do not exclude
samples on the basis of the quantity you are testing. A test the deposit cannot
support is reported as `not_testable`, never quietly replaced by one it can.
{revision}

{data_access}

## Datasets available

{datasets_block}

## Your task

1. Check that the deposit can test the claim before analyzing anything. Find
   the folders that hold the claim's outcome measurement and confirm they
   measure it directly, for the samples, conditions and replicates the claim
   needs. A staining panel is not a functional assay; a folder that covers 3
   clones cannot test a 12-clone claim. When the Pennsieve tools are
   available, call `describe_discover_folder` on each folder you plan to use.
   If the deposit cannot test the claim as stated, do not substitute a proxy:
   write a short notebook that documents the check, and report the status
   `not_testable` (step 6).
   Then inspect the data itself. Look at actual columns, dtypes, NaN rates —
   do NOT assume schema from the title alone.
2. Before computing any result, write `{PLAN_FILE}`, as JSON, setting down the
   test you are about to run, taken from the pre-specified plan above:
   `{{"primary_comparison": "<the one comparison that decides the claim>",
     "arms": ["<every group or condition the claim names>"],
     "metric": "<the quantity compared, as the prediction states it>",
     "planned_exclusions": ["<any, with the control-or-QC reason>"],
     "statistical_test": "<the test and what counts as the independent unit>",
     "deviations": []}}`
   Write it before you look at any outcome, and do not edit those fields
   afterwards. If the data force a change, append an entry to `deviations`:
   `{{"changed": "<what>", "why": "<the reason>", "when": "<before or after
   seeing the result>"}}`. A deviation found after seeing the result is still
   reportable — declaring it is what keeps the analysis honest; hiding it in a
   revised plan is what does not.
3. Write a Jupyter notebook (`analysis.ipynb`) that tests the claim
   end-to-end: load → preprocess → analysis → conclusion. Report the
   pre-specified primary comparison and every arm the claim names, including
   the arms whose result argues against the claim. Anything beyond the plan is
   welcome but labelled exploratory, and never replaces the primary result.
4. Run it via papermill. Fix errors iteratively.
   Save the executed output as `analysis.ipynb`; every nonempty code cell
   must have an execution count. Never substitute synthetic data for the
   deposit. If the data cannot support the test, report the blocker.
5. When the notebook runs cleanly, write a short summary of what you
   found and whether the claim is supported, partially supported, or
   not supported by the data. Be honest about assumptions and limits.
6. Write `{STATUS_FILE}` in your working directory, as JSON:
   `{{"status": "tested", "data_files": ["data/files/...", ...]}}` when the
   notebook analyzed the deposit's data, listing every data file it read.
   Each entry is a path relative to your working directory. A glob such as
   `data/files/DATA/sub-*/derivatives/electrodes.tsv` is fine when you read
   many files of the same shape, as long as it matches files that are really
   there. Or
   `{{"status": "not_testable", "reason": "<what the deposit lacks>", "data_files": [...]}}`
   when the deposit cannot test the claim as stated (step 1), listing the
   files you checked, or
   `{{"status": "blocked", "blocker": "<what stopped you>", "data_files": []}}`
   when you could not reach or read the data. Both are valid, honest
   outcomes. Never report "tested" for an analysis that did not read the
   deposit's data.

Your working directory is `{workdir}`. Write all files there.
"""


def _format_datasets_brief(bb: Blackboard) -> str:
    if not bb.dataset_summaries:
        return "(no datasets on the blackboard — Hypothesize did not record any)"
    out = []
    for d in bb.dataset_summaries:
        out.append(
            f"- **{d.id}**: {d.title}\n"
            f"  - {d.description[:300]}"
            f"\n  - Repository URL: {d.url}"
            f"\n  - README: {d.readme[:1500]}"
            f"\n  - File tree: {d.file_tree[:10000]}"
            f"\n  - Record tables: {json.dumps(d.record_tables)[:3000]}"
        )
    return "\n".join(out)


STATUS_FILE = "analysis_status.json"
PLAN_FILE = "analysis_plan.json"


class NotTestable(RuntimeError):
    """The deposit cannot test the hypothesis as stated. An honest outcome,
    recorded on the hypothesis; it is not an analysis failure."""


class AnalystRefused(RuntimeError):
    """The session's model refused the task (stop_reason "refusal"). The
    pipeline reruns the round on a fallback model (`VP_ANALYZE_FALLBACK_MODEL`)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"{model_id} refused the analysis task")

PENNSIEVE_TOOLS_SECTION = """## Pennsieve tools

You also have Pennsieve tools for published datasets. They are public and need no login:

- `find_discover_files`: search every file of a dataset by path pattern
  (`*` within a folder, `**` across folders) or file type. Use it instead of
  opening folders one by one.
- `describe_discover_folder`: what the FCS or EDF files in one folder record
  (channel layouts, stains or sampling rates, event counts) and the distinct
  sample names in it. Use it to check that a planned test's channels,
  conditions and samples exist before writing the notebook.
- `summarize_discover_table`: the columns, guessed types and first rows of a
  CSV or TSV file.
- `list_discover_files`: the files in one folder of a dataset. Each file has a
  `download_path`; pass it to the other tools.
- `read_discover_file_headers`: what an FCS or EDF file records (channel names
  and stains, or signal labels and sampling rates), without downloading it.
  Use it to check that the channels or signals your test needs exist.
- `preview_discover_file`: the first part of a text file, such as a TSV or CSV.
- `get_discover_file_link`: a short-lived download link for one file.
- `get_discover_dataset`, `search_discover_datasets`, `lookup_citation`: dataset
  metadata, other published datasets, and DOI lookups.

Download the files your notebook analyzes into `data/` with the download
command above, so the notebook can be run again.

Run that download in the foreground and wait for it, even when it takes
minutes. Do not put it in the background and poll it with `sleep`: that spends
your turns and your time budget on waiting. The download skips files already
present at their published size, so if it is cut short, running the same
command again resumes it rather than starting over."""


PLAN_FIELDS = ("primary_comparison", "arms", "metric", "statistical_test")


def check_analysis_plan(workdir: Path) -> dict:
    """Require the analyst to have set down the test before running it.

    The pre-specified prediction and experiment come from Hypothesize and were
    what its critic accepted. Writing them down first, and declaring any
    departure as a deviation, is what separates troubleshooting an analysis
    from redesigning it around the result. A deviation is allowed; an
    undeclared one is not, so an empty or reasonless entry fails here.
    """
    path = workdir / PLAN_FILE
    if not path.is_file():
        raise RuntimeError(f"the analyst did not write {PLAN_FILE}, so the test it ran "
                           f"cannot be compared with the one that was approved")
    try:
        plan = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(f"{PLAN_FILE} is not valid JSON: {exc}") from exc
    empty = [f for f in PLAN_FIELDS if not plan.get(f)]
    if empty:
        raise RuntimeError(f"{PLAN_FILE} does not say: {', '.join(empty)}")
    deviations = plan.get("deviations") or []
    if not isinstance(deviations, list):
        raise RuntimeError(f"{PLAN_FILE} has a 'deviations' that is not a list")
    unexplained = [d for d in deviations
                   if not isinstance(d, dict) or not d.get("changed") or not d.get("why")]
    if unexplained:
        raise RuntimeError(f"{PLAN_FILE} has {len(unexplained)} deviation(s) with no "
                           f"'changed' and 'why'; an undeclared change to the plan is "
                           f"the one thing the record cannot carry")
    return plan


def check_analysis_status(workdir: Path) -> dict:
    """Require the analyst's own report that it analyzed the deposit's data.

    A notebook can run cleanly and still analyze nothing, for example when it
    only reports that the data could not be downloaded. The analyst writes
    `analysis_status.json`; this raises unless the status is "tested" and every
    data file it lists exists in the working directory.
    """
    path = workdir / STATUS_FILE
    if not path.is_file():
        raise RuntimeError(f"the analyst did not write {STATUS_FILE}")
    try:
        status = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(f"{STATUS_FILE} is not valid JSON: {exc}") from exc
    if status.get("status") == "not_testable":
        raise NotTestable(status.get("reason") or "(no reason given)")
    if status.get("status") == "blocked":
        raise RuntimeError(f"the analyst reported the analysis as blocked: "
                           f"{status.get('blocker') or '(no reason given)'}")
    if status.get("status") != "tested":
        raise RuntimeError(f"{STATUS_FILE} has status {status.get('status')!r}; expected 'tested'")
    files = status.get("data_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"{STATUS_FILE} lists no data files that the analysis read")
    missing = [f for f in files if not _entry_matches_a_file(workdir, f)]
    if missing:
        raise RuntimeError(f"{STATUS_FILE} lists data files that are not in {workdir}: "
                           + ", ".join(map(str, missing[:5])))
    return status


# A trailing note after a path, such as "sub-*/electrodes.tsv (all 29 subjects)".
_TRAILING_NOTE = re.compile(r"\s*\([^()]*\)\s*$")


def _entry_matches_a_file(workdir: Path, entry: str) -> bool:
    """Does one `data_files` entry name at least one real file under `workdir`?

    The check exists so the analyst cannot claim data it never read, so every
    entry must still resolve to a file that is really there. The entry itself
    may be written in any form an analyst reasonably reaches for: a plain path,
    a glob covering many subjects, or either of those followed by a note in
    brackets. A run on `pennsieve:530` failed on
    `data/files/DATA/sub-*/derivatives/electrodes.tsv (all 29 subjects)`, which
    was a true statement about 29 files that were present.
    """
    text = _TRAILING_NOTE.sub("", str(entry)).strip()
    if not text:
        return False
    root = workdir.resolve()

    def inside_and_real(path: Path) -> bool:
        resolved = path.resolve()
        return resolved.is_relative_to(root) and resolved.is_file()

    if any(ch in text for ch in "*?["):
        return any(inside_and_real(match) for match in workdir.glob(text))
    return inside_and_real(workdir / text)


# Fields only a code cell may carry. An analyst that edits notebook JSON directly can turn a
# code cell into markdown and leave these behind, which nbformat's schema rejects.
CODE_ONLY_CELL_FIELDS = ("execution_count", "outputs")


def tidy_notebook(path: Path) -> list[int]:
    """Drop code-only fields from non-code cells, in place, and return the cells changed.

    The fields mean nothing on a markdown or raw cell, so removing them changes no
    code, output or result. Without this, a harmless leftover from editing the JSON
    fails schema validation and discards an otherwise finished analysis.
    """
    import json

    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code" and any(k in cell for k in CODE_ONLY_CELL_FIELDS):
            for key in CODE_ONLY_CELL_FIELDS:
                cell.pop(key, None)
            changed.append(i)
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def validate_notebook(path: Path) -> None:
    """Require actual executed cells; absence of errors alone is insufficient."""
    import nbformat

    tidied = tidy_notebook(path)
    if tidied:
        print(f"  [analyze] {path.name}: removed code-only fields from non-code cells {tidied}",
              flush=True)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    if not code or any(c.execution_count is None for c in code):
        raise RuntimeError("analysis.ipynb has no executed code or contains unexecuted cells")
    if any(o.output_type == "error" for c in code for o in c.outputs):
        raise RuntimeError("analysis.ipynb contains execution errors")
