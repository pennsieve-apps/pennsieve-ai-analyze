# pennsieve-ai-analyze

Stage 3 of the `pennsieve-ai-*` research pipeline. Reads the blackboard
produced by [`pennsieve-ai-hypothesize`](https://github.com/Pennsieve/pennsieve-ai-hypothesize),
runs an agentic analysis against the first hypothesis in the final iteration
(or first N, configurable), and emits an updated blackboard for the downstream
Write stage.

## What it does

For each selected hypothesis:

1. **Agentic Analyst** — a Claude Code session with Read/Write/Edit/Bash
   tools. The agent inspects the dataset(s) named in the hypothesis,
   writes a Jupyter notebook that tests the claim, runs it via
   papermill, and iterates on errors. Returns when the notebook runs
   cleanly or the turn budget is exhausted.

2. **Methods Critic** — a single-shot reasoning-role call that reviews the
   analyst's summary + notebook and flags methodological issues. Output
   is a structured JSON verdict attached to the hypothesis.

At startup, Analyze discovers the governor's effective allow-list and verifies
that its concrete assignments are invokable. The Agent SDK analyst requests the
logical synthesis role; the methods critic requests the reasoning role. With
the allow-list used for v0.0.3 validation, both resolve to Sonnet 4.6. A
mid-run `model_not_allowed` or IAM `access_denied` response reroutes and retries
the direct methods-critic request up to three bounded attempts.

The platform colocates a `llm-governor-proxy` sidecar in every
LLM-enabled ECS task, so the user container doesn't see any AWS auth
plumbing. The Claude Agent SDK reads `ANTHROPIC_BASE_URL=http://localhost:8765`
from env and routes all its API traffic through the sidecar, which
bridges to the governor via `lambda:Invoke[WithResponseStream]`.

## Run locally

Requires:
- `LLM_GOVERNOR_FUNCTION_NAME` for model discovery and direct critic calls
- `ANTHROPIC_BASE_URL` pointed at a running `llm-governor-proxy` instance for
  the Claude Agent SDK analyst
- `INPUT_DIR` with a `blackboard.json` produced by Hypothesize
- `OUTPUT_DIR` writable

```bash
pip install -e .

INPUT_DIR=/tmp/hypothesize-output \
OUTPUT_DIR=/tmp/analyze-output \
EXECUTION_RUN_ID=local-dev \
LLM_GOVERNOR_FUNCTION_NAME=your-governor-lambda-name \
ANTHROPIC_BASE_URL=http://localhost:8765 \
python -m analyze.main
```

## Config

| Env var | Default | Purpose |
|---|---|---|
| `INPUT_DIR` | `/mnt/efs/input` | Where Hypothesize wrote `blackboard.json` |
| `OUTPUT_DIR` | `/mnt/efs/output` | Where we write updated blackboard + per-hypothesis workdirs |
| `EXECUTION_RUN_ID` | `local-dev` | Platform-injected run id (cost attribution) |
| `WORKFLOW_INSTANCE_ID` | `local-dev` | Platform-injected workflow execution identifier |
| `LLM_GOVERNOR_FUNCTION_NAME` | _(required)_ | Governor Lambda used for discovery, preflight, and direct critic calls |
| `ANTHROPIC_BASE_URL` | _(required)_ | Sidecar URL (`http://localhost:8765` in production) |
| `VP_ANALYZE_TOP_N` | `1` | How many hypotheses from the final iteration to analyze |
| `VP_ANALYZE_MAX_TURNS` | `80` | Per-hypothesis turn budget for the Claude Code loop |
| `VP_MODEL_SONNET` | dynamically resolved | Explicit pin for the Agent SDK analyst's model role |
| `VP_ANALYZE_MODEL` | unset | Backward-compatible alias for `VP_MODEL_SONNET`; conflicting pins fail preflight |
| `VP_MODEL_OPUS` | dynamically resolved | Explicit pin for the methods critic's reasoning role |
| `VP_LLM_MAX_RETRIES` | `1` | Anthropic SDK retries within each direct routed attempt |

## Pennsieve Test smoke check

Before the analyst starts, a healthy run prints the effective allow-list and
both concrete assignments. For the allow-list that motivated v0.0.3, the
important lines are:

```text
--- LLM preflight ---
  assignments:
    opus    (reasoning) -> us.anthropic.claude-sonnet-4-6 [ranked]
    sonnet  (analysis)  -> us.anthropic.claude-sonnet-4-6 [ranked]

[agent] hypothesis H1: model=us.anthropic.claude-sonnet-4-6 ...
[llm] → methods_critic logical=opus model=us.anthropic.claude-sonnet-4-6 attempt=1/3 ...
```

Successful completion should log `methods_critic: done` followed by
`analyze: done`. In the output `blackboard.json`, the selected hypothesis from
the final Hypothesize iteration must contain non-empty `analysis_workdir` and
`analysis_summary` fields plus a structured `methods_critique`. Those are the
handoff fields required by Write.

## Output

Written to `$OUTPUT_DIR`:

- `blackboard.json` — updated blackboard. Each analyzed hypothesis now
  carries `analysis_workdir`, `analysis_summary`, `analysis_cost_usd`,
  `analysis_tools_called`, `analysis_started_at`/`finished_at`, and
  `methods_critique` fields.
- `hypotheses/{hypothesis_id}/` — per-hypothesis working directory
  containing the agent's notebook, data files, intermediate outputs.
- `notebook.jsonl` — append-only audit log of every blackboard mutation
  (one event per line).
- `usage.jsonl` — token-usage records for the non-Claude-Code agent
  calls (methods critic). Claude Code's own cost is on the hypothesis
  in `analysis_cost_usd`.

Analyze treats the analyst and methods critic as required phases. A failure in
either is logged, persisted to the output audit files, and returned as a failed
stage instead of being reported as `analyze: done`.

## Blackboard compatibility

Hypothesize keeps one hypothesis record per iteration, including locked copies
with the same ID. Analyze deliberately updates the final-iteration record so
Write's latest-iteration selection reads the same object. Upstream `degraded`
and `model_assignments` fields survive loading; this release then records
Analyze's stage-local assignments in `model_assignments`. Cumulative,
stage-namespaced provenance remains deferred to the shared
`pennsieve-ai-core` blackboard contract.

## Status

**v0.0.3 candidate — locally validated, not yet tested in Pennsieve compute.**
