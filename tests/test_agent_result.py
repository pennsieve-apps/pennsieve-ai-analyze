import asyncio
from types import SimpleNamespace

import claude_agent_sdk
import nbformat
import pytest

from analyze.agents import agentic_analyst
from analyze.blackboard import Blackboard, Hypothesis


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
