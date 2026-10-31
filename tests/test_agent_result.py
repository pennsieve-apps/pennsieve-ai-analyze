import asyncio
import json
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
    (workdir / agentic_analyst.PLAN_FILE).write_text(json.dumps({
        "primary_comparison": "treated vs untreated", "arms": ["treated", "untreated"],
        "metric": "background-subtracted gMFI", "planned_exclusions": [],
        "statistical_test": "paired t-test, replicate as the unit", "deviations": []}))


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


@pytest.mark.parametrize("where", ["assistant", "result"])
def test_a_refusal_in_the_session_raises_analyst_refused(monkeypatch, tmp_path, where):
    seen = {}

    class Client:
        def __init__(self, options):
            seen["fallback"] = options.fallback_model
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            pass
        async def receive_response(self):
            if where == "assistant":
                yield SimpleNamespace(model="m", content=[], stop_reason="refusal")
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.2, result="",
                                  usage={}, stop_reason="refusal" if where == "result" else "end_turn")
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    monkeypatch.delenv("VP_ANALYZE_FALLBACK_MODEL", raising=False)
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    with pytest.raises(agentic_analyst.AnalystRefused, match="us.anthropic.claude-opus-5-5"):
        asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="us.anthropic.claude-opus-5-5"))
    assert h.analysis_cost_usd == 0.2
    # Claude Code's own fallback is set too, for an overloaded or unavailable model.
    assert seen["fallback"] == agentic_analyst.DEFAULT_ANALYST_FALLBACKS[0]


def test_the_fallback_list_skips_the_model_already_in_use(monkeypatch):
    monkeypatch.delenv("VP_ANALYZE_FALLBACK_MODEL", raising=False)
    chain = list(agentic_analyst.DEFAULT_ANALYST_FALLBACKS)
    assert agentic_analyst.analyst_fallbacks(chain[0]) == chain[1:]
    assert agentic_analyst.analyst_fallbacks("x", can_invoke=lambda m: m == chain[2]) == [chain[2]]
    monkeypatch.setenv("VP_ANALYZE_FALLBACK_MODEL", "a, b")
    assert agentic_analyst.analyst_fallbacks("a") == ["b"]
    monkeypatch.setenv("VP_ANALYZE_FALLBACK_MODEL", "none")
    assert agentic_analyst.analyst_fallbacks("x") == []


def test_a_cancelled_session_still_records_what_it_spent(monkeypatch, tmp_path):
    """The SDK reports usage only on its result message. A session killed by
    the hypothesis timeout spent real money; without this it reaches no usage
    log and the run's cost is understated. Seed 405's H1 cost $0.60 that way.
    """
    class Client:
        def __init__(self, options):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def query(self, prompt):
            pass
        async def receive_response(self):
            for _ in range(3):
                yield SimpleNamespace(content=[], usage={"input_tokens": 10, "output_tokens": 5,
                                                         "cache_read_input_tokens": 900,
                                                         "cache_creation_input_tokens": 100})
            raise asyncio.CancelledError
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    usage_log = []

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="test",
                                        usage_log=usage_log))

    assert len(usage_log) == 1
    row = usage_log[0]
    assert row["partial"] is True and row["turns"] == 3
    assert row["input_tokens"] == 30 and row["output_tokens"] == 15
    assert row["cache_read_input_tokens"] == 2700
    # The cost roll-up counts it like any other analyst row.
    from pennsieve_ai_utils import pricing
    assert pricing.summarize_usage(usage_log)["requests"] == 1


def test_a_completed_session_records_one_row_only(monkeypatch, tmp_path):
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
            yield SimpleNamespace(content=[], usage={"input_tokens": 10, "output_tokens": 5})
            yield SimpleNamespace(subtype="success", is_error=False, total_cost_usd=0.1,
                                  result="summary", usage={"input_tokens": 10, "output_tokens": 5})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", Client)
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "prediction", "experiment")
    usage_log = []
    asyncio.run(agentic_analyst.run(bb, h, workdir=tmp_path, model_id="test", usage_log=usage_log))
    assert len(usage_log) == 1 and "partial" not in usage_log[0]


# --- holding the analysis to the plan that was approved --------------------
#
# Hypothesize writes a predicted observation and an experiment, and its critic
# accepts the hypothesis on them. Until now neither reached Analyze, so the
# analyst designed its own test — and in seed 405 it dropped an arm the claim
# named, swapped the metric, and excluded samples on the quantity under test.


def _plan(**overrides):
    plan = {"primary_comparison": "treated vs untreated", "arms": ["treated", "untreated"],
            "metric": "background-subtracted gMFI", "planned_exclusions": [],
            "statistical_test": "paired t-test, replicate as the unit", "deviations": []}
    plan.update(overrides)
    return plan


def test_the_approved_test_reaches_the_analyst(tmp_path):
    bb = Blackboard("q", "dataset")
    h = Hypothesis("H1", "claim", "rationale", [], "H15K6 falls more than CD227 and H4K11",
                   "stain all three clones, compare gMFI ratios")
    prompt = agentic_analyst._build_task_prompt(bb, h, tmp_path)
    assert "H15K6 falls more than CD227 and H4K11" in prompt
    assert "stain all three clones" in prompt
    assert "do not drop an arm the claim names" in prompt


def test_an_analysis_with_no_recorded_plan_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="did not write analysis_plan.json"):
        agentic_analyst.check_analysis_plan(tmp_path)


@pytest.mark.parametrize("missing", ["primary_comparison", "arms", "metric", "statistical_test"])
def test_a_plan_that_omits_the_test_is_refused(tmp_path, missing):
    (tmp_path / agentic_analyst.PLAN_FILE).write_text(json.dumps(_plan(**{missing: ""})))
    with pytest.raises(RuntimeError, match=missing):
        agentic_analyst.check_analysis_plan(tmp_path)


def test_a_declared_deviation_is_allowed(tmp_path):
    """Changing the plan is permitted; the record just has to carry it."""
    plan = _plan(deviations=[{"changed": "dropped the H4K11 arm",
                              "why": "untreated signal below the detection floor",
                              "when": "after seeing the result"}])
    (tmp_path / agentic_analyst.PLAN_FILE).write_text(json.dumps(plan))
    assert agentic_analyst.check_analysis_plan(tmp_path)["deviations"][0]["changed"]


def test_a_deviation_with_no_reason_is_refused(tmp_path):
    (tmp_path / agentic_analyst.PLAN_FILE).write_text(
        json.dumps(_plan(deviations=[{"changed": "dropped the H4K11 arm"}])))
    with pytest.raises(RuntimeError, match="no 'changed' and 'why'"):
        agentic_analyst.check_analysis_plan(tmp_path)
