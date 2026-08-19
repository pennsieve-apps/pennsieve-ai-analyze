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
auth ourselves — the platform handles it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from analyze.blackboard import Blackboard
from analyze.llm import LLMClient
from analyze.pipeline import run_analyze


def main() -> int:
    input_dir = Path(os.environ.get("INPUT_DIR", "/mnt/efs/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/mnt/efs/output"))
    execution_run_id = os.environ.get("EXECUTION_RUN_ID", "local-dev")
    workflow_instance_id = os.environ.get("WORKFLOW_INSTANCE_ID", "local-dev")

    print("=== pennsieve-ai-analyze ===")
    print(f"  INPUT_DIR={input_dir}")
    print(f"  OUTPUT_DIR={output_dir}")
    print(f"  EXECUTION_RUN_ID={execution_run_id}")
    print(f"  WORKFLOW_INSTANCE_ID={workflow_instance_id}")
    print(f"  ANTHROPIC_BASE_URL={os.environ.get('ANTHROPIC_BASE_URL', '<unset>')}")

    # Sanity-check the platform contract: if ANTHROPIC_BASE_URL isn't set,
    # the Claude Agent SDK would default to api.anthropic.com (with no
    # API key set) and fail in a confusing way. Fail fast with a clear
    # message instead — this almost certainly means the workflow was
    # configured without LLM access enabled, OR the sidecar didn't start.
    if not os.environ.get("ANTHROPIC_BASE_URL"):
        print(
            "ERROR: ANTHROPIC_BASE_URL is not set. The analyze stage requires "
            "the LLM governor sidecar (platform-managed). Confirm the compute "
            "node has `enable_llm_access = true`, or check the sidecar's "
            "container logs (stream prefix: <node-id>-proxy/llm-governor-proxy/...).",
            file=sys.stderr,
        )
        return 2

    bb_path = input_dir / "blackboard.json"
    if not bb_path.exists():
        print(
            f"ERROR: no blackboard.json at {bb_path} — Hypothesize did not "
            f"produce its expected output, or the workflow wiring is wrong.",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- loading blackboard from {bb_path} ---")
    bb = Blackboard.load(bb_path)
    bb.run_id = execution_run_id
    if bb.degraded:
        stages = ", ".join(sorted({d.get("stage", "?") for d in bb.degraded}))
        print(f"  WARNING: upstream blackboard is degraded — skipped/failed stages: {stages}")
        bb.log("analyze", "upstream_degraded", stages)
    bb.log("analyze", "start",
           f"loaded blackboard: {len(bb.dataset_summaries)} datasets, "
           f"{len(bb.hypotheses)} hypotheses, "
           f"{len(bb.literature_claims)} literature claims")
    print(f"  research_question: {bb.research_question[:200]}")
    print(f"  hypotheses: {len(bb.hypotheses)} | "
          f"datasets: {len(bb.dataset_summaries)} | "
          f"literature claims: {len(bb.literature_claims)}")

    # Non-Claude-Code agents (e.g., the methods critic) still use a plain
    # anthropic.Anthropic client. The Claude Agent SDK in the analyst path
    # is independent — it reads ANTHROPIC_BASE_URL from env directly.
    client = LLMClient()

    try:
        bb.model_assignments = client.preflight()
        bb.log(
            "analyze", "models",
            ", ".join(
                f"{logical}={assignment['model_id']}"
                for logical, assignment in sorted(bb.model_assignments.items())
            ),
        )
        asyncio.run(run_analyze(bb, client, workdir=output_dir))
    except Exception as e:
        bb.log("analyze", "fatal", repr(e))
        _write_outputs(bb, client, output_dir)
        raise

    bb.log("analyze", "done", "")
    _write_outputs(bb, client, output_dir)
    print(f"\nWrote: {output_dir / 'blackboard.json'}")
    return 0


def _write_outputs(bb: Blackboard, client: LLMClient, output_dir: Path) -> None:
    """Persist blackboard + audit logs to OUTPUT_DIR for the Write stage."""
    bb.save(output_dir / "blackboard.json")
    nb = "\n".join(json.dumps(e) for e in bb.notebook)
    (output_dir / "notebook.jsonl").write_text(nb + "\n" if nb else "")
    ul = "\n".join(json.dumps(e) for e in client.usage_log)
    (output_dir / "usage.jsonl").write_text(ul + "\n" if ul else "")


if __name__ == "__main__":
    raise SystemExit(main())
