import asyncio
from types import SimpleNamespace

import claude_agent_sdk
import nbformat
import pytest

from analyze.agents import agentic_analyst
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis


@pytest.mark.parametrize("subtype,notebook,passes", [("success", True, True), ("error_max_turns", True, False), ("success", False, False)])
def test_agent_requires_success_result_and_executed_notebook(monkeypatch, tmp_path, subtype, notebook, passes):
    class Client:
        def __init__(self, options):
            assert options.allowed_tools == []
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            if notebook:
                cell = nbformat.v4.new_code_cell("print(1)", execution_count=1)
                nbformat.write(nbformat.v4.new_notebook(cells=[cell]), tmp_path / "hypotheses/H1/analysis.ipynb")
        async def receive_response(self):
            yield SimpleNamespace(subtype=subtype, is_error=subtype != "success", total_cost_usd=0.1, result="analysis summary", usage={})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    if passes:
        asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="test"))
        assert h.analysis_summary == "analysis summary"
    else:
        with pytest.raises(RuntimeError, match="Agentic analysis failed"):
            asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="test"))
        assert h.analysis_summary == ""
    assert h.analysis_cost_usd == 0.1
    assert '"event": "result"' in (tmp_path / "hypotheses/H1/agent-events.jsonl").read_text()


def test_raw_io_trace_keeps_the_full_transcript(monkeypatch, tmp_path):
    import json

    long_output = "x" * (agentic_analyst.TRACE_RESULT_CHARS + 10)

    class Client:
        def __init__(self, options):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            cell = nbformat.v4.new_code_cell("print(1)", execution_count=1)
            nbformat.write(nbformat.v4.new_notebook(cells=[cell]), tmp_path / "hypotheses/H1/analysis.ipynb")
        async def receive_response(self):
            yield SimpleNamespace(model="m", usage={"input_tokens": 5, "output_tokens": 1}, content=[
                SimpleNamespace(thinking="plan", signature=""), SimpleNamespace(text="looking")])
            yield SimpleNamespace(content=[SimpleNamespace(tool_use_id="t1", content=long_output)])
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.1, result="done", usage={})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    monkeypatch.setenv("VP_LOG_LLM_IO", "1")
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="test"))
    rows = [json.loads(line) for line in (tmp_path / "hypotheses/H1/agent-events.jsonl").read_text().splitlines()]
    kinds = [r["event"] for r in rows]
    assert kinds == ["start", "thinking", "text", "turn_usage", "tool_result", "result"]
    assert "claim" in rows[0]["prompt"]
    assert rows[4]["content"] == long_output
