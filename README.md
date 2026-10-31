# pennsieve-ai-analyze

Stage 3 of the `pennsieve-ai-*` research pipeline. Reads the blackboard
produced by [`pennsieve-ai-hypothesize`](https://github.com/Pennsieve/pennsieve-ai-hypothesize),
runs an agentic analysis against the best-ranked accepted hypothesis in the final
iteration (or the best N, configurable), and emits an updated blackboard for the
downstream Write stage.

Accepted hypotheses are ranked by the critic's scores: feasibility first, then
novelty, then the sum of all four scores. Fewer `[CONSTRAINT]` or `[READOUT]`
attacks break a tie, and Hypothesize's list order breaks what is left. The
ranking is logged as `analyze: selection`.

If a chosen hypothesis turns out not testable, or its methods review is still
unresolved after the last round, Analyze moves on to the next-ranked one and
logs `analyze: fallback`. It stops once `VP_ANALYZE_TOP_N` have passed or
`VP_ANALYZE_MAX_ATTEMPTS` have been tried.

## What it does

For each selected hypothesis:

0. **Data staging** — before the agent starts, the seed dataset's file index
   and a bounded sample are downloaded into the hypothesis working directory
   from the public Pennsieve Discover route. The analyst then works from local
   files instead of spending turns probing an API. A dataset whose index
   cannot be read raises an error, so the stage stops instead of guessing.
   The sample size is bounded by `VP_ANALYZE_STAGE_FILES` and
   `VP_ANALYZE_STAGE_MAX_MB`; set `VP_ANALYZE_STAGE_DATA=0` to write the index
   only. (This currently runs on every backend; it assumes the deposit is
   published to Discover and reachable from the compute node.)

1. **Agentic Analyst** — a Claude Code session with Read/Write/Edit/Bash
   tools, and the `pennsieve-mcp` Discover tools when the run provides one.
   The agent inspects the dataset(s) named in the hypothesis,
   writes a Jupyter notebook that tests the claim, runs it via
   papermill, and iterates on errors. The stage requires a successful SDK
   terminal result and a cleanly executed notebook; turn-budget exhaustion is
   retained in the trace and fails the analysis. It reads FCS files with
   `flowio`.

2. **Analyst ⇄ Methods Critic loop** — the reasoning-role critic reviews the
   analyst's summary and notebook. A `concerns` or `unsound` verdict, or any
   high-severity issue, sends the critique back through another analyst pass.
   The bounded loop succeeds only on a `sound` verdict without high-severity
   issues; each notebook and critique is snapshotted for review.

**Honest non-pass outcomes.** Not every hypothesis can be tested with the
deposit, and that is a result rather than a crash. When the only remaining
high-severity issues are data limits the analysis cannot fix, the stage
records the hypothesis as **not testable** and exits 7. When the methods
review simply never passes within the allowed rounds, it records the
hypothesis as **unresolved** and exits 8. Both write the reason to
`not_testable.json` or `unresolved.json` in the hypothesis folder, keep the
notebooks, and do not count as a pass, so Write builds no manuscript on them.

**The approved test is the test that runs.** Hypothesize records a predicted
observation and a suggested experiment, and its critic accepts the hypothesis
on them. Both are given to the analyst as the plan to follow, and to the
methods critic as the standard to judge against. Before computing any result
the analyst writes `analysis_plan.json` — the primary comparison, the arms the
claim names, the metric, the planned exclusions and the statistical test — and
any later change goes in `deviations` with a reason. Troubleshooting (code,
gates, file handling) is expected; dropping an arm, swapping the metric,
narrowing the comparison, excluding samples on the quantity under test, or
choosing a multiplicity family after seeing results is a `plan_deviation`
issue at high severity unless it was declared. That sends the hypothesis back
for a revision round rather than letting the analysis settle around its own
result.

**Budgets bound a hypothesis, not the stage.** A wall-clock cap on the whole
stage fires in the middle of a hypothesis and takes the finished ones down with
it, leaving a run nobody can interpret. Instead each hypothesis has its own
time budget (`VP_ANALYZE_HYPOTHESIS_TIMEOUT_S`): past it the hypothesis is
recorded as **timed out** and the next one starts. A run cost ceiling
(`VP_RUN_COST_LIMIT_USD`) stops the stage between hypotheses, where everything
finished is already on disk, and exits 9.

**Every hypothesis is checkpointed.** The moment one settles — passed, not
testable, unresolved, timed out or refused — its outcome and the fields Write
reads are written to `hypotheses/<id>/outcome.json`. Running the stage again
skips what is already settled, so a retry pays only for the undecided ones.
An attempt that died mid-analysis leaves no outcome, so its folder is moved to
`<id>.attempt<n>` and the hypothesis is analyzed again on a clean directory.
To retry one that did settle, delete its folder.

**Refusals.** If the analyst's model refuses the task, Analyze reruns that
round on `VP_ANALYZE_FALLBACK_MODEL` (Opus 4.7 by default) and keeps that model
for the rest of the stage. It logs `analyze: refusal_fallback`. If every model
refuses, the hypothesis is recorded as **refused** (`refused.json`) and the
next-ranked hypothesis is tried. If none passes, the stage fails. The methods
critic's refusals are handled by the model client (see `VP_MODEL_REASONING_FALLBACK`
in the pennsieve-ai-utils README).

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
| `VP_ANALYZE_TOP_N` | `all` | How many of the best-ranked accepted hypotheses from the final iteration to analyze. `all` tests every accepted hypothesis, as the prototype did. A number tests only the best-ranked ones; each that passes is written up, and the rest are recorded as not testable or unresolved. |
| `VP_ANALYZE_MAX_ATTEMPTS` | all accepted, or `VP_ANALYZE_TOP_N` + 1 | The most hypotheses Analyze will try, counting fallbacks. Each try is a full analyst session, so this caps the spending. Set it equal to `VP_ANALYZE_TOP_N` to turn the fallback off. |
| `VP_ANALYZE_HYPOTHESIS_IDS` | unset | A comma-separated list of hypothesis IDs to analyze, in order, instead of taking the best-ranked `VP_ANALYZE_TOP_N`. It overrides the ranking, and there is no fallback. Each ID must be accepted at the final iteration. Use it to reach a specific accepted hypothesis when the top-ranked one is not testable. |
| `VP_ANALYZE_MAX_TURNS` | `80` | Per-hypothesis turn budget for the Claude Code loop |
| `VP_ANALYZE_MAX_METHODS_ITERS` | `3` | Maximum analyst/critic passes before an unresolved methods review fails the stage |
| `VP_ANALYZE_HYPOTHESIS_TIMEOUT_S` | `5400` | How long one hypothesis may take, across all its rounds. Past it the hypothesis is recorded as `timed_out` and the next one starts, so a stuck analyst costs one hypothesis rather than the stage. `off` removes the cap. |
| `VP_RUN_COST_LIMIT_USD` | unset | A ceiling on what the whole run may cost, counting the earlier stages. It is checked between hypotheses, where everything finished is already on disk, and the stage exits 9. Raise it and run the stage again to continue. |
| `VP_ANALYZE_RESUME` | `1` | Skip hypotheses whose outcome is already recorded in `hypotheses/<id>/outcome.json`, so a retry pays only for the undecided ones. Set to `0` to analyze everything again. To retry one hypothesis, delete its folder. |
| `VP_ANALYZE_STAGE_FILES` | `12` | How many of the seed dataset's data files to download before the analyst starts |
| `VP_ANALYZE_STAGE_MAX_MB` | `200` | Total size budget, in MiB, for that pre-download |
| `VP_ANALYZE_STAGE_DATA` | unset | Set to `0` to write only the file index and download nothing |
| `VP_MODEL_SONNET` | dynamically resolved | Explicit pin for the Agent SDK analyst's model role |
| `VP_ANALYZE_MODEL` | unset | Backward-compatible alias for `VP_MODEL_SONNET`; conflicting pins fail preflight |
| `VP_ANALYZE_FALLBACK_MODEL` | `us.anthropic.claude-opus-4-7` | Comma-separated models the analyst moves to when its model refuses the task. Also passed to Claude Code's own fallback for an overloaded or unavailable model. `none` turns it off. |
| `VP_MODEL_OPUS` | dynamically resolved | Explicit pin for the methods critic's reasoning role |
| `VP_LLM_MAX_RETRIES` | `1` on the governor, `4` on `bedrock` or `anthropic` | How many times the Anthropic SDK retries a failed request. The governor keeps one retry, because each of its timeouts already takes a minute. Direct backends retry more, to ride out short network dropouts. |

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
