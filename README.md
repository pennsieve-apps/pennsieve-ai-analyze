# pennsieve-ai-analyze

Stage 3 of the `pennsieve-ai-*` research pipeline. Reads the blackboard
produced by [`pennsieve-ai-hypothesize`](https://github.com/Pennsieve/pennsieve-ai-hypothesize),
runs an agentic analysis against the top-ranked hypothesis (or top-N
configurable), and emits an updated blackboard for the downstream Write
stage.

## What it does

For each selected hypothesis:

1. **Agentic Analyst** — a Claude Code session with Read/Write/Edit/Bash
   tools. The agent inspects the dataset(s) named in the hypothesis,
   writes a Jupyter notebook that tests the claim, runs it via
   papermill, and iterates on errors. Returns when the notebook runs
   cleanly or the turn budget is exhausted.

2. **Methods Critic** — a single-shot Opus call that reviews the
   analyst's summary + notebook and flags methodological issues. Output
   is a structured JSON verdict attached to the hypothesis.

The platform colocates a `llm-governor-proxy` sidecar in every
LLM-enabled ECS task, so the user container doesn't see any AWS auth
plumbing. The Claude Agent SDK reads `ANTHROPIC_BASE_URL=http://localhost:8765`
from env and routes all its API traffic through the sidecar, which
bridges to the governor via `lambda:Invoke[WithResponseStream]`.

## Run locally

Requires:
- `LLM_GOVERNOR_FUNCTION_NAME` (or `ANTHROPIC_BASE_URL` pointed at a
  running `llm-governor-proxy` instance)
- `INPUT_DIR` with a `blackboard.json` produced by Hypothesize
- `OUTPUT_DIR` writable

```bash
pip install -e .

INPUT_DIR=/tmp/hypothesize-output \
OUTPUT_DIR=/tmp/analyze-output \
EXECUTION_RUN_ID=local-dev \
ANTHROPIC_BASE_URL=http://localhost:8765 \
python -m analyze.main
```

## Config

| Env var | Default | Purpose |
|---|---|---|
| `INPUT_DIR` | `/mnt/efs/input` | Where Hypothesize wrote `blackboard.json` |
| `OUTPUT_DIR` | `/mnt/efs/output` | Where we write updated blackboard + per-hypothesis workdirs |
| `EXECUTION_RUN_ID` | `local-dev` | Platform-injected run id (cost attribution) |
| `ANTHROPIC_BASE_URL` | _(required)_ | Sidecar URL (`http://localhost:8765` in production) |
| `VP_ANALYZE_TOP_N` | `1` | How many top hypotheses to analyze in this run |
| `VP_ANALYZE_MAX_TURNS` | `80` | Per-hypothesis turn budget for the Claude Code loop |
| `VP_ANALYZE_MODEL` | Sonnet 4.5 | Bedrock inference-profile ID the analyst uses |

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
