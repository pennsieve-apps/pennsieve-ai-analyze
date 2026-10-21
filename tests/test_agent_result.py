import asyncio
from types import SimpleNamespace

import claude_agent_sdk
import nbformat
import pytest

from analyze.agents import agentic_analyst
from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis


@pytest.fixture(autouse=True)
def _no_staging(monkeypatch):
    """These tests cover the agent loop; data staging has its own tests."""
    monkeypatch.setattr(agentic_analyst.data_staging, "stage", lambda bb, h, workdir: None)


def write_outputs(workdir):
    """What a successful analyst leaves behind: an executed notebook, the data
    file it read, and its status report."""
    cell = nbformat.v4.new_code_cell("print(1)", execution_count=1)
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), workdir / "analysis.ipynb")
    (workdir / "data").mkdir(exist_ok=True)
    (workdir / "data/a.fcs").write_text("x")
    (workdir / agentic_analyst.STATUS_FILE).write_text(
        '{"status": "tested", "data_files": ["data/a.fcs"]}')


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
                write_outputs(tmp_path / "hypotheses/H1")
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
            write_outputs(tmp_path / "hypotheses/H1")
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


def test_analyst_tokens_are_added_to_the_usage_log(monkeypatch, tmp_path):
    class Client:
        def __init__(self, options):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            write_outputs(tmp_path / "hypotheses/H1")
        async def receive_response(self):
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.5, result="done",
                                  num_turns=7, usage={"input_tokens": 10, "output_tokens": 20,
                                                      "cache_read_input_tokens": 300,
                                                      "cache_creation_input_tokens": 40})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    usage = []
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="m", usage_log=usage))
    row = usage[-1]
    assert row["event"] == "agent_sdk_result" and row["hypothesis_id"] == "H1" and row["turns"] == 7
    assert (row["input_tokens"], row["output_tokens"]) == (10, 20)
    assert (row["cache_read_input_tokens"], row["cache_creation_input_tokens"]) == (300, 40)
    assert row["sdk_reported_cost_usd"] == 0.5 and "cost_usd" not in row


def test_a_clean_notebook_that_analyzed_no_data_fails(monkeypatch, tmp_path):
    class Client:
        def __init__(self, options):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            write_outputs(tmp_path / "hypotheses/H1")
            (tmp_path / "hypotheses/H1" / agentic_analyst.STATUS_FILE).write_text(
                '{"status": "blocked", "blocker": "403 from S3", "data_files": []}')
        async def receive_response(self):
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.1,
                                  result="blocked", usage={})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    with pytest.raises(RuntimeError, match="Agentic analysis failed"):
        asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="m"))
    assert h.analysis_summary == ""


def test_the_analyst_gets_the_pennsieve_tools_when_the_run_provides_them(monkeypatch, tmp_path):
    seen = {}

    class Client:
        def __init__(self, options):
            seen["options"] = options
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            seen["prompt"] = prompt
            write_outputs(tmp_path / "hypotheses/H1")
        async def receive_response(self):
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.1, result="done", usage={})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    monkeypatch.setenv("VP_PENNSIEVE_MCP_URL", "http://pennsieve-mcp:8090/mcp")
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="m"))
    options = seen["options"]
    assert options.mcp_servers == {"pennsieve": {"type": "http", "url": "http://pennsieve-mcp:8090/mcp"}}
    assert "read_discover_file_headers" in seen["prompt"]
    allow = asyncio.run(options.can_use_tool("mcp__pennsieve__list_discover_files", {}, None))
    deny = asyncio.run(options.can_use_tool("mcp__other__anything", {}, None))
    assert type(allow).__name__ == "PermissionResultAllow" and type(deny).__name__ == "PermissionResultDeny"


def test_without_a_server_the_analyst_gets_no_mcp_tools(monkeypatch, tmp_path):
    seen = {}

    class Client:
        def __init__(self, options):
            seen["options"] = options
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            write_outputs(tmp_path / "hypotheses/H1")
        async def receive_response(self):
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.1, result="done", usage={})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    monkeypatch.delenv("VP_PENNSIEVE_MCP_URL", raising=False)
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    asyncio.run(agentic_analyst.run(Blackboard("q", "dataset"), h, workdir=tmp_path, model_id="m"))
    assert seen["options"].mcp_servers == {}
    deny = asyncio.run(seen["options"].can_use_tool("mcp__pennsieve__list_discover_files", {}, None))
    assert type(deny).__name__ == "PermissionResultDeny"
