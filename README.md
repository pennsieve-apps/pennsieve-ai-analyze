# pennsieve-ai-analyze

Stage 3 of the `pennsieve-ai-*` research pipeline. Reads the blackboard
produced by [`pennsieve-ai-hypothesize`](https://github.com/Pennsieve/pennsieve-ai-hypothesize),
runs an agentic analysis against the first accepted hypothesis in the final iteration
(or first N, configurable), and emits an updated blackboard for the downstream
Write stage.

## What it does

For each selected hypothesis:

1. **Agentic Analyst** — a Claude Code session with Read/Write/Edit/Bash
   tools. The agent inspects the dataset(s) named in the hypothesis,
   writes a Jupyter notebook that tests the claim, runs it via
   papermill, and iterates on errors. The stage requires a successful SDK
   terminal result and a cleanly executed notebook; turn-budget exhaustion is
   retained in the trace and fails the analysis.

2. **Analyst ⇄ Methods Critic loop** — the reasoning-role critic reviews the
   analyst's summary and notebook. A `concerns` or `unsound` verdict, or any
   high-severity issue, sends the critique back through another analyst pass.
   The bounded loop succeeds only on a `sound` verdict without high-severity
   issues; each notebook and critique is snapshotted for review.

At startup, Analyze asks the governor which models it allows. It then sends a
tiny test request to each model it plans to use. The analyst uses the
synthesis role. That role uses Sonnet 5.5, or Sonnet 4.6 if 5.5 is not allowed.
The methods critic uses the reasoning role. That role uses Opus 5.5, then
another Opus such as Opus 4.7, then Sonnet 5.5. If no allowed model is strong
enough for a role, the stage stops at startup.

Sometimes the governor refuses a model partway through a run. When that
happens, the methods critic picks the next allowed model and tries again. It
tries at most three times.

The platform colocates a `llm-governor-proxy` sidecar in every
LLM-enabled ECS task, so the user container doesn't see any AWS auth
plumbing. The Claude Agent SDK reads `ANTHROPIC_BASE_URL=http://localhost:8765`
from env and routes all its API traffic through the sidecar, which
bridges to the governor via `lambda:Invoke[WithResponseStream]`.

The Agent SDK runs the Claude Code program inside this container. Claude Code
never uses a Claude subscription or anyone's Claude login. It signs in only
with settings it finds in the environment. On the platform, it goes through the
sidecar. When you run it yourself, it uses your own AWS profile or API key, as
described below.

## Run locally

Requires:
- A way to reach the models. The `VP_LLM_BACKEND` setting chooses it. The
  [pennsieve-ai-utils README](https://github.com/Pennsieve/pennsieve-ai-utils#llm-client)
  explains the choices.
  - With `governor`, the default, set `LLM_GOVERNOR_FUNCTION_NAME`. The stage
    uses it to find the allowed models and to call the methods critic. Also set
    `ANTHROPIC_BASE_URL` to a running `llm-governor-proxy`. The analyst's Claude
    Code session sends its requests there.
  - With `bedrock`, set `AWS_REGION` and an AWS profile or AWS keys. The stage
    then tells Claude Code to use Bedrock by itself. Claude Code also makes
    small requests in the background. The stage sends those to the `bulk`
    model.
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

# Or call Bedrock directly. Name the model for every role.
INPUT_DIR=/tmp/hypothesize-output \
OUTPUT_DIR=/tmp/analyze-output \
VP_LLM_BACKEND=bedrock AWS_PROFILE=pennsieve-bedrock AWS_REGION=us-east-1 \
VP_MODEL_REASONING=us.anthropic.claude-opus-5-5 \
VP_MODEL_SYNTHESIS=us.anthropic.claude-sonnet-5-5 \
VP_MODEL_BULK=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
python -m analyze.main
```

## Config

| Env var | Default | Purpose |
|---|---|---|
| `INPUT_DIR` | `/mnt/efs/input` | Where Hypothesize wrote `blackboard.json` |
| `OUTPUT_DIR` | `/mnt/efs/output` | Where we write updated blackboard + per-hypothesis workdirs |
| `EXECUTION_RUN_ID` | `local-dev` | Platform-injected run id (cost attribution) |
| `WORKFLOW_INSTANCE_ID` | `local-dev` | Platform-injected workflow execution identifier |
| `VP_LLM_BACKEND` | `governor` | This chooses where model requests go. Use `bedrock` or `anthropic` only for local testing. |
| `LLM_GOVERNOR_FUNCTION_NAME` | _(required for `governor`)_ | Governor Lambda used for discovery, preflight, and direct critic calls |
| `ANTHROPIC_BASE_URL` | _(required for `governor`)_ | Sidecar URL (`http://localhost:8765` in production) |
| `AWS_REGION` | _(required for `bedrock`)_ | This is the Bedrock region. Claude Code does not read the region from your AWS profile, so you must set it here. |
| `VP_MODEL_BULK` | chosen at startup | On `bedrock` or `anthropic`, Claude Code uses this model for its small background requests. |
| `VP_LOG_LLM_IO` | unset | Set this to `1` to save model requests and responses word for word. The methods critic's requests go to `llm-io.analyze.jsonl`. The file `agent-events.jsonl` becomes a full record of the Claude Code session. |
| `VP_ANALYZE_TOP_N` | `1` | How many accepted hypotheses from the final iteration to analyze |
| `VP_ANALYZE_MAX_TURNS` | `80` | Per-hypothesis turn budget for the Claude Code loop |
| `VP_ANALYZE_MAX_METHODS_ITERS` | `3` | Maximum analyst/critic passes before an unresolved methods review fails the stage |
| `VP_MODEL_SONNET` | dynamically resolved | Explicit pin for the Agent SDK analyst's model role |
| `VP_ANALYZE_MODEL` | unset | Backward-compatible alias for `VP_MODEL_SONNET`; conflicting pins fail preflight |
| `VP_MODEL_OPUS` | dynamically resolved | Explicit pin for the methods critic's reasoning role |
| `VP_LLM_MAX_RETRIES` | `1` | Anthropic SDK retries within each direct routed attempt |

## Pennsieve Test smoke check

Before the analyst starts, a healthy run prints the effective allow-list and
the model chosen for each role. If the governor allows the 5.5 models, the
important lines look like this:

```text
--- LLM preflight ---
  role assignments:
    reasoning  → us.anthropic.claude-opus-5-5 (rank 355)  [ranked]
    synthesis  → us.anthropic.claude-sonnet-5-5 (rank 255)  [ranked]

[agent] hypothesis H1: model=us.anthropic.claude-sonnet-5-5 ...
[llm] → methods_critic  role=opus  model=us.anthropic.claude-opus-5-5 ...
```

Successful completion may include several analyst/critic passes. It logs
`analyze: done` only after the methods gate passes. In the output
`blackboard.json`, the selected hypothesis from
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
  containing the agent's notebook, data files, intermediate outputs,
  `agent-events.jsonl`, and paired `analysis.iter{N}.ipynb` /
  `methods.iter{N}.json` snapshots.
- `blackboard.{hypothesis_id}.iter{N}.json` — blackboard snapshot after each
  methods-review pass.
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
Write's latest-iteration selection reads the same object. The shared
`pennsieve-ai-utils` contract provides schema migrations and preserves unknown
fields. Analyze retains upstream model provenance and records its assignments
under the `analyze` stage namespace.

## Status

**v0.0.6 release candidate — locally validated; Pennsieve Test rerun pending.**
This candidate restricts analysis to accepted final-round hypotheses, enforces
successful SDK and notebook completion, and adds a bounded methods-revision
gate with reviewable traces and snapshots. It requires the Anthropic
1.x-compatible Governor transport from `pennsieve-llm` 0.6.1. The rerun will
confirm the current Bedrock assignments.
