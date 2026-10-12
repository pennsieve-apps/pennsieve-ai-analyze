"""Entrypoint for pennsieve-ai-analyze.

Reads Hypothesize's blackboard from `$INPUT_DIR/blackboard.json`, runs
the agentic-analyst loop against one or more hypotheses, and writes the
updated blackboard (now carrying analysis artifacts) to `$OUTPUT_DIR`
for the Write stage.

Auth model: the platform colocates an `llm-governor-proxy` sidecar in
every LLM-enabled ECS task and injects `ANTHROPIC_BASE_URL=http://localhost:8765`
into this container's env. The Claude Agent SDK reads that env var and
sends all its API traffic at the sidecar, which bridges to the governor
via lambda:Invoke. We don't touch AWS creds, SigV4, or any Anthropic
auth ourselves — the platform handles it. Local runs can bypass the governor
with `VP_LLM_BACKEND=bedrock` or `anthropic` (see `pennsieve_ai_utils.backend`).
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace

from pennsieve_ai_utils.backend import llm_backend, missing_config
from pennsieve_ai_utils.llm import LLMClient
from pennsieve_ai_utils.models import ROLES, ModelNotAvailable
from pennsieve_ai_utils.processor import (
    ProcessorEnv,
    install_sigterm_handler,
    load_upstream_blackboard,
    write_outputs,
)

from analyze.pipeline import run_analyze

STAGE = "analyze"

# Analyze predates the shared role names; `VP_ANALYZE_MODEL` pins the analyst
# (synthesis) model on existing deployments and keeps working as an alias.
ROLE_TABLE = {
    **ROLES,
    "synthesis": replace(ROLES["synthesis"], env_aliases=("VP_ANALYZE_MODEL",)),
}


def main() -> int:
    install_sigterm_handler()
    env = ProcessorEnv.from_environ()
    backend = llm_backend()
    env.banner(STAGE, VP_LLM_BACKEND=backend,
               ANTHROPIC_BASE_URL=os.environ.get("ANTHROPIC_BASE_URL", "<unset>"))

    # Sanity-check the backend contract. On the platform a missing
    # ANTHROPIC_BASE_URL would send the Claude Agent SDK to api.anthropic.com
    # (with no API key set) and fail in a confusing way. Fail fast with a
    # clear message instead — this almost certainly means the workflow was
    # configured without LLM access enabled, OR the sidecar didn't start.
    missing = missing_config(claude_code=True)
    if missing:
        hint = (
            "The analyze stage requires the LLM governor sidecar (platform-managed). "
            "Confirm the compute node has `enable_llm_access = true`, or check the "
            "sidecar's container logs (stream prefix: "
            "<node-id>-proxy/llm-governor-proxy/...)."
            if backend == "governor" else
            f"Set them for VP_LLM_BACKEND={backend}."
        )
        print(f"ERROR: {', '.join(missing)} not set. {hint}", file=sys.stderr)
        return 2

    bb = load_upstream_blackboard(env.input_dir / "blackboard.json", stage=STAGE,
                                  upstream="Hypothesize", run_id=env.execution_run_id)
    if bb is None:
        return 2
    output_dir = env.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bb.log(STAGE, "start",
           f"loaded blackboard: {len(bb.dataset_summaries)} datasets, "
           f"{len(bb.hypotheses)} hypotheses, "
           f"{len(bb.literature_claims)} literature claims")
    print(f"  research_question: {bb.research_question[:200]}")
    print(f"  hypotheses: {len(bb.hypotheses)} | "
          f"datasets: {len(bb.dataset_summaries)} | "
          f"literature claims: {len(bb.literature_claims)}")
    bb.save(output_dir / "blackboard.json")

    # Non-Claude-Code agents (the methods critic) use the governor client
    # directly. The Claude Agent SDK in the analyst path is independent — it
    # reads ANTHROPIC_BASE_URL from env — but takes its model ID from the same
    # preflighted assignment (`synthesis`).
    # A direct backend also preflights `bulk`, which serves Claude Code's
    # background calls there.
    roles = ("reasoning", "synthesis") + (("bulk",) if backend != "governor" else ())
    client = LLMClient(stage=STAGE, roles=roles, role_table=ROLE_TABLE)
    try:
        assignments = client.preflight()
    except ModelNotAvailable as e:
        bb.log(STAGE, "fatal", f"model preflight failed: {e}")
        write_outputs(bb, output_dir, client.usage_log, stage=STAGE)
        print(f"ERROR: {e}", file=sys.stderr)
        return 5
    for role, assignment in assignments.items():
        bb.record_model_assignment(stage=STAGE, role=role, assignment=assignment)
    bb.log(STAGE, "models",
           ", ".join(f"{role}={a['model_id']}" for role, a in sorted(assignments.items())))

    try:
        asyncio.run(run_analyze(bb, client, workdir=output_dir))
    except (Exception, KeyboardInterrupt) as e:
        bb.log(STAGE, "fatal", repr(e))
        write_outputs(bb, output_dir, client.usage_log, stage=STAGE)
        raise

    bb.log(STAGE, "done", "")
    write_outputs(bb, output_dir, client.usage_log, stage=STAGE)
    print(f"\nWrote: {output_dir / 'blackboard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
