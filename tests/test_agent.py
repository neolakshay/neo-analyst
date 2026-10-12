# tests/test_agent.py
"""
Tests for Phase 2, Step 1: Core LLM-powered data analysis agent.

All tests use a mocked OpenAI client and run completely offline without requiring
a real API key or making network requests.
"""

from pathlib import Path
from unittest.mock import MagicMock
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from openai import OpenAIError

from app.agent import (
    AnalysisAgent,
    AgentResult,
    ToolCallRecord,
    TOOL_SCHEMAS,
    execute_tool,
)
from app.agent.agent import build_dataset_metadata, _parse_tool_arguments, _validate_tool_call_entry
from app.agent.tools import to_json_safe, REGISTERED_TOOL_NAMES
from app.analysis import group_and_aggregate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Evan"],
        "department": ["Engineering", "HR", "Engineering", "Marketing", "HR"],
        "salary": [85000.0, 52000.0, 91000.0, 68000.0, 54000.0],
        "bonus": [12000.0, np.nan, 15000.0, 8000.0, np.nan],
        "rating": [4.8, 3.9, 4.5, 4.1, 3.5],
    })


def make_tool_call(call_id: str, name: str, arguments: dict | str):
    """Helper to create a mocked tool call object matching OpenAI structure."""
    if isinstance(arguments, dict):
        arg_str = json.dumps(arguments)
    else:
        arg_str = str(arguments)

    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = arg_str
    return tc


def make_completion(content: str = None, tool_calls: list = None, finish_reason: str = None):
    """Helper to create a mocked ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── 1. Tool Schemas and Registration ──────────────────────────────────────────

def test_tool_schemas_completeness():
    """Ensure all six analytical functions are registered with proper schemas."""
    expected_tools = {
        "get_dataset_overview",
        "get_summary_statistics",
        "filter_data",
        "analyze_correlations",
        "generate_visualization",
        "group_and_aggregate",
    }
    assert set(REGISTERED_TOOL_NAMES) == expected_tools
    assert len(TOOL_SCHEMAS) == 6

    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] in expected_tools
        assert "description" in fn and len(fn["description"]) > 10
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


# ── 2. Tool Dispatch Tests ───────────────────────────────────────────────────

def test_execute_get_dataset_overview(sample_df):
    model_out, raw_out = execute_tool(sample_df, "get_dataset_overview", {})
    assert model_out["rows"] == 5
    assert model_out["columns"] == 6
    assert "salary" in model_out["column_names"]
    assert "salary" in model_out["dtypes"]
    assert "missing_counts" in model_out
    assert isinstance(raw_out, dict)


def test_execute_get_summary_statistics(sample_df):
    model_out, raw_out = execute_tool(
        sample_df, "get_summary_statistics", {"top_n": 3}
    )
    assert "numeric" in model_out
    assert "salary" in model_out["numeric"]
    assert model_out["numeric"]["salary"]["mean"] == 70000.0
    assert "categorical" in model_out
    assert isinstance(raw_out, dict)


def test_execute_filter_data(sample_df):
    args = {
        "conditions": [{"column": "salary", "operator": "gte", "value": 80000}]
    }
    model_out, raw_out = execute_tool(sample_df, "filter_data", args, preview_rows=2)
    assert model_out["status"] == "success"
    assert model_out["matched_rows"] == 2
    assert model_out["total_rows"] == 5
    assert model_out["preview_rows"] == 2
    assert len(model_out["preview"]) == 2
    assert isinstance(raw_out, pd.DataFrame)
    assert len(raw_out) == 2


def test_execute_analyze_correlations(sample_df):
    args = {"columns": ["salary", "rating"], "top_n": 2}
    model_out, raw_out = execute_tool(sample_df, "analyze_correlations", args)
    assert "matrix" in model_out
    assert "top_positive" in model_out
    assert "pair_counts" in model_out
    assert model_out["pair_counts"]["salary"]["rating"] == 5
    assert isinstance(raw_out, dict)


def test_execute_generate_visualization(sample_df):
    args = {
        "chart_type": "bar",
        "x_column": "department",
        "y_column": "salary",
        "title": "Salary by Dept",
    }
    model_out, raw_out = execute_tool(sample_df, "generate_visualization", args)
    assert model_out["status"] == "success"
    assert model_out["chart_type"] == "bar"
    assert model_out["title"] == "Salary by Dept"
    assert isinstance(raw_out, go.Figure)


def test_execute_group_and_aggregate(sample_df):
    args = {
        "group_by": ["department"],
        "aggregations": {"salary": ["mean", "count"]},
    }
    model_out, raw_out = execute_tool(sample_df, "group_and_aggregate", args)
    assert "results" in model_out
    assert isinstance(model_out["results"], list)
    dept_names = [item["department"] for item in model_out["results"]]
    assert "Engineering" in dept_names
    eng_record = next(item for item in model_out["results"] if item["department"] == "Engineering")
    assert eng_record["salary_mean"] == 88000.0
    assert isinstance(raw_out, dict)


def test_execute_unknown_tool(sample_df):
    model_out, raw_out = execute_tool(sample_df, "run_arbitrary_python", {"code": "1+1"})
    assert model_out["status"] == "error"
    assert "Unknown tool" in model_out["error"]
    assert raw_out is None


def test_execute_tool_failure(sample_df):
    # Invalid column name in filter_data
    args = {"conditions": [{"column": "non_existent", "operator": "eq", "value": 10}]}
    model_out, raw_out = execute_tool(sample_df, "filter_data", args)
    assert model_out["status"] == "error"
    assert "non_existent" in model_out["error"]
    assert raw_out is None


# ── 3. Serialization and Missing Value Handling ──────────────────────────────

def test_to_json_safe_scalars_and_structures():
    data = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "np_int": np.int64(42),
        "np_float": np.float64(3.14159),
        "np_bool": np.bool_(True),
        "pd_na": pd.NA,
        "nested": {"items": [1, np.nan, "hello"]},
    }
    cleaned = to_json_safe(data)
    # Must be JSON-dumpable
    dumped = json.dumps(cleaned)
    assert dumped is not None
    assert cleaned["nan_val"] is None
    assert cleaned["inf_val"] is None
    assert cleaned["np_int"] == 42
    assert cleaned["np_float"] == 3.1416
    assert cleaned["np_bool"] is True
    assert cleaned["pd_na"] is None
    assert cleaned["nested"]["items"] == [1, None, "hello"]


def test_build_dataset_metadata(sample_df):
    meta = build_dataset_metadata(sample_df, preview_rows=2)
    assert meta["rows"] == 5
    assert len(meta["columns"]) == 6
    assert meta["missing_counts"] == {"bonus": 2}
    assert len(meta["preview_sample"]) == 2


# ── 4. Agent Tool-Calling Cycle (Mocked) ──────────────────────────────────────

def test_agent_complete_cycle(sample_df):
    """Test full cycle: user query -> tool call -> tool result -> final answer."""
    mock_client = MagicMock()
    # Round 1: Model calls get_dataset_overview
    tc = make_tool_call("call_1", "get_dataset_overview", {})
    resp1 = make_completion(tool_calls=[tc])
    # Round 2: Model gives final text answer
    resp2 = make_completion(content="The dataset has 5 rows and 6 columns.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Give me an overview of the dataset.")

    assert isinstance(result, AgentResult)
    assert result.answer == "The dataset has 5 rows and 6 columns."
    assert result.rounds == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_dataset_overview"
    assert result.tool_calls[0].output["rows"] == 5
    assert result.error is None
    assert mock_client.chat.completions.create.call_count == 2


def test_agent_multiple_tool_calls_in_single_round(sample_df):
    """Test when the model invokes multiple tools in a single response round."""
    mock_client = MagicMock()
    tc1 = make_tool_call("call_1", "get_dataset_overview", {})
    tc2 = make_tool_call(
        "call_2",
        "group_and_aggregate",
        {"group_by": ["department"], "aggregations": {"salary": ["mean"]}},
    )
    resp1 = make_completion(tool_calls=[tc1, tc2])
    resp2 = make_completion(content="Engineering has the highest average salary.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Compare departments and dataset size.")

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "get_dataset_overview"
    assert result.tool_calls[1].tool_name == "group_and_aggregate"
    assert result.answer == "Engineering has the highest average salary."
    assert result.rounds == 2


def test_agent_multi_round_tool_calls(sample_df):
    """Test multi-round reasoning: tool -> tool -> answer."""
    mock_client = MagicMock()
    tc1 = make_tool_call(
        "call_1",
        "filter_data",
        {"conditions": [{"column": "department", "operator": "eq", "value": "HR"}]},
    )
    resp1 = make_completion(tool_calls=[tc1])

    tc2 = make_tool_call(
        "call_2",
        "generate_visualization",
        {"chart_type": "bar", "x_column": "name", "y_column": "salary"},
    )
    resp2 = make_completion(tool_calls=[tc2])

    resp3 = make_completion(content="Created the HR salary chart.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Show me a chart of HR salaries.")

    assert result.rounds == 3
    assert len(result.tool_calls) == 2
    assert len(result.figures) == 1
    assert isinstance(result.figures[0], go.Figure)
    assert result.answer == "Created the HR salary chart."


def test_agent_preserves_plotly_figures(sample_df):
    """Ensure Plotly figures are kept in result.figures and not serialized into string garbage."""
    mock_client = MagicMock()
    tc = make_tool_call(
        "call_viz",
        "generate_visualization",
        {"chart_type": "box", "y_column": "salary"},
    )
    resp1 = make_completion(tool_calls=[tc])
    resp2 = make_completion(content="Here is the box plot.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Box plot of salaries.")

    assert len(result.figures) == 1
    assert isinstance(result.figures[0], go.Figure)


def test_agent_bounded_preview(sample_df):
    """Ensure filter_data produces a bounded preview."""
    mock_client = MagicMock()
    tc = make_tool_call(
        "call_filt",
        "filter_data",
        {"conditions": [{"column": "salary", "operator": "gt", "value": 0}]},
    )
    resp1 = make_completion(tool_calls=[tc])
    resp2 = make_completion(content="All 5 salaries are positive.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Find all positive salaries.")

    tool_rec = result.tool_calls[0]
    assert tool_rec.output["matched_rows"] == 5
    assert len(tool_rec.output["preview"]) <= 5


def test_agent_does_not_mutate_input_dataframe(sample_df):
    """Ensure original DataFrame is unmodified after agent execution."""
    df_copy = sample_df.copy(deep=True)
    mock_client = MagicMock()
    tc = make_tool_call(
        "call_1",
        "filter_data",
        {"conditions": [{"column": "salary", "operator": "gt", "value": 60000}]},
    )
    resp1 = make_completion(tool_calls=[tc])
    resp2 = make_completion(content="Filtered successfully.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    agent.run(sample_df, "Filter high earners.")

    pd.testing.assert_frame_equal(sample_df, df_copy)


# ── 5. Error Handling and Edge Cases ──────────────────────────────────────────

def test_agent_invalid_arguments_json(sample_df):
    """Handle model returning invalid JSON for tool arguments."""
    mock_client = MagicMock()
    # Pass malformed JSON as arguments string
    tc = make_tool_call("call_broken", "filter_data", "NOT_VALID_JSON{")
    resp1 = make_completion(tool_calls=[tc])
    resp2 = make_completion(content="Recovered from invalid arguments.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Filter with broken json.")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].output["status"] == "error"
    assert "Invalid JSON" in result.tool_calls[0].output["error"]
    assert result.answer == "Recovered from invalid arguments."


def test_agent_max_rounds_enforcement(sample_df):
    """Loop terminates when maximum rounds is reached without infinite recursion."""
    mock_client = MagicMock()
    # Always return a tool call
    tc = make_tool_call("call_loop", "get_dataset_overview", {})
    resp_loop = make_completion(tool_calls=[tc])
    mock_client.chat.completions.create.return_value = resp_loop

    agent = AnalysisAgent(client=mock_client, max_rounds=3)
    result = agent.run(sample_df, "Infinite loop query.")

    assert result.rounds == 3
    assert len(result.tool_calls) == 3
    assert result.error is not None
    assert "Exhausted maximum tool-calling rounds" in result.error
    # Max-rounds termination must NOT produce a fabricated success answer
    assert result.answer == ""


def test_agent_openai_api_error(sample_df):
    """Gracefully handle OpenAI API exception."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = OpenAIError("Rate limit exceeded")

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Test error handling.")

    assert result.error is not None
    assert "OpenAI API error: Rate limit exceeded" in result.error
    assert result.rounds == 1


def test_agent_input_validation(sample_df):
    """Test parameter validation for DataFrame and question."""
    agent = AnalysisAgent(client=MagicMock())

    # Invalid dataframe type
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        agent.run([1, 2, 3], "Question")

    # Empty columns dataframe
    with pytest.raises(ValueError, match="no columns"):
        agent.run(pd.DataFrame(), "Question")

    # Invalid question type
    with pytest.raises(TypeError, match="Question must be a string"):
        agent.run(sample_df, 12345)

    # Empty question
    with pytest.raises(ValueError, match="Question cannot be empty"):
        agent.run(sample_df, "   ")


def test_agent_missing_api_key(monkeypatch):
    """Ensure clear error when no API key is available."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = AnalysisAgent(api_key=None, client=None)
    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        agent.run(pd.DataFrame({"a": [1]}), "Analyze this")


def test_agent_result_to_dict():
    """Verify AgentResult serialization via to_dict."""
    tc = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={"rows": 10},
        raw_output=None,
    )
    fig = go.Figure()
    res = AgentResult(
        answer="Done",
        tool_calls=[tc],
        figures=[fig],
        rounds=1,
        model="gpt-4o-mini",
        error=None,
    )
    d = res.to_dict()
    assert d["answer"] == "Done"
    assert d["figures_count"] == 1
    assert len(d["tool_calls"]) == 1
    assert d["tool_calls"][0]["tool_name"] == "get_dataset_overview"
    assert d["model"] == "gpt-4o-mini"
    assert d["error"] is None


# ── 6. Response Structure Hardening ──────────────────────────────────────────

def test_agent_empty_choices_returns_error(sample_df):
    """Empty choices list in API response produces a structured error, not an exception."""
    mock_client = MagicMock()
    bad_resp = MagicMock()
    bad_resp.choices = []  # Empty list — would cause IndexError without guard
    mock_client.chat.completions.create.return_value = bad_resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Any question.")

    assert result.error is not None
    assert "choices" in result.error.lower() or "empty" in result.error.lower()
    assert result.answer == ""
    assert result.rounds == 1


def test_agent_none_choices_returns_error(sample_df):
    """None choices in API response produces a structured error, not an exception."""
    mock_client = MagicMock()
    bad_resp = MagicMock()
    bad_resp.choices = None
    mock_client.chat.completions.create.return_value = bad_resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Any question.")

    assert result.error is not None
    assert result.answer == ""
    assert result.rounds == 1


def test_agent_none_message_returns_error(sample_df):
    """None message on a choice produces a structured error, not an exception."""
    mock_client = MagicMock()
    bad_choice = MagicMock()
    bad_choice.message = None
    bad_resp = MagicMock()
    bad_resp.choices = [bad_choice]
    mock_client.chat.completions.create.return_value = bad_resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Any question.")

    assert result.error is not None
    assert "message" in result.error.lower() or "none" in result.error.lower()
    assert result.answer == ""
    assert result.rounds == 1


def test_agent_tool_calls_before_error_are_preserved(sample_df):
    """Tool-call records and figures accumulated before a structural error are not discarded."""
    mock_client = MagicMock()
    # Round 1: valid tool call
    tc = make_tool_call("call_1", "get_dataset_overview", {})
    resp1 = make_completion(tool_calls=[tc])
    # Round 2: malformed response (empty choices)
    bad_resp = MagicMock()
    bad_resp.choices = []
    mock_client.chat.completions.create.side_effect = [resp1, bad_resp]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Overview then fail.")

    assert result.error is not None
    assert len(result.tool_calls) == 1  # The first successful tool call is still present
    assert result.tool_calls[0].tool_name == "get_dataset_overview"


# ── 7. Argument Parsing Unit Tests ───────────────────────────────────────────

def test_parse_tool_arguments_valid_dict():
    """Pre-parsed dict passes through unchanged."""
    args, err = _parse_tool_arguments({"top_n": 3}, "get_summary_statistics")
    assert err is None
    assert args == {"top_n": 3}


def test_parse_tool_arguments_valid_json_string():
    """Valid JSON object string is decoded to dict."""
    args, err = _parse_tool_arguments('{"group_by": ["dept"]}', "group_and_aggregate")
    assert err is None
    assert args == {"group_by": ["dept"]}


def test_parse_tool_arguments_empty_object():
    """Empty JSON object '{}' is valid."""
    args, err = _parse_tool_arguments("{}", "get_dataset_overview")
    assert err is None
    assert args == {}


def test_parse_tool_arguments_malformed_json():
    """Malformed JSON string produces a structured error dict."""
    args, err = _parse_tool_arguments("NOT_VALID{", "filter_data")
    assert err is not None
    assert err["status"] == "error"
    assert "Invalid JSON" in err["error"]
    assert args == {}


def test_parse_tool_arguments_json_array_is_rejected():
    """A JSON array (not object) is invalid tool arguments."""
    args, err = _parse_tool_arguments('["a", "b"]', "filter_data")
    assert err is not None
    assert err["status"] == "error"
    assert "JSON object" in err["error"]
    assert args == {}


def test_parse_tool_arguments_json_scalar_is_rejected():
    """A JSON scalar (number, string) is invalid tool arguments."""
    args, err = _parse_tool_arguments("42", "filter_data")
    assert err is not None
    assert err["status"] == "error"
    assert args == {}


def test_parse_tool_arguments_none_type_is_rejected():
    """None arguments produce a structured error."""
    args, err = _parse_tool_arguments(None, "get_dataset_overview")
    assert err is not None
    assert err["status"] == "error"
    assert "unexpected type" in err["error"]
    assert args == {}


def test_agent_non_dict_json_args_sends_error_to_model(sample_df):
    """When the model sends a JSON array as args, error is returned to it for recovery."""
    mock_client = MagicMock()
    # Model sends a JSON array instead of an object
    tc = make_tool_call("call_bad", "get_dataset_overview", '["wrong", "args"]')
    resp1 = make_completion(tool_calls=[tc])
    resp2 = make_completion(content="Recovered from bad args.")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Send non-dict args.")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].output["status"] == "error"
    assert "JSON object" in result.tool_calls[0].output["error"]
    assert result.answer == "Recovered from bad args."


# ── 8. finish_reason Handling ────────────────────────────────────────────────

def test_finish_reason_stop_is_normal_answer(sample_df):
    """finish_reason='stop' with no tool_calls → accepted as a final text answer."""
    mock_client = MagicMock()
    resp = make_completion(content="Done.", finish_reason="stop")
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Simple question.")

    assert result.answer == "Done."
    assert result.error is None
    assert result.rounds == 1


def test_finish_reason_tool_calls_dispatches_tools(sample_df):
    """finish_reason='tool_calls' triggers tool dispatch then final answer."""
    mock_client = MagicMock()
    tc = make_tool_call("call_1", "get_dataset_overview", {})
    resp1 = make_completion(tool_calls=[tc], finish_reason="tool_calls")
    resp2 = make_completion(content="Overview done.", finish_reason="stop")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Give me an overview.")

    assert result.error is None
    assert result.answer == "Overview done."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_dataset_overview"


def test_finish_reason_length_produces_error(sample_df):
    """finish_reason='length' produces a structured error; answer is not set."""
    mock_client = MagicMock()
    resp = make_completion(content="Truncated...", finish_reason="length")
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Long question.")

    assert result.error is not None
    assert "length" in result.error
    assert "truncated" in result.error.lower() or "token" in result.error.lower()
    assert result.answer == ""
    assert result.rounds == 1


def test_finish_reason_content_filter_produces_error(sample_df):
    """finish_reason='content_filter' produces a structured error; answer is not set."""
    mock_client = MagicMock()
    resp = make_completion(content=None, finish_reason="content_filter")
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Sensitive query.")

    assert result.error is not None
    assert "content_filter" in result.error
    assert "blocked" in result.error.lower() or "safety" in result.error.lower()
    assert result.answer == ""
    assert result.rounds == 1


def test_finish_reason_none_with_no_tool_calls_treated_as_answer(sample_df):
    """Missing finish_reason (None) with no tool_calls falls back to capturing text."""
    mock_client = MagicMock()
    resp = make_completion(content="Best-effort answer.", finish_reason=None)
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Question with no finish_reason.")

    # None finish_reason with no tool_calls → captured as answer, no hard error
    assert result.answer == "Best-effort answer."
    assert result.rounds == 1


def test_finish_reason_unknown_with_no_tool_calls_captures_content_with_notice(sample_df):
    """Unknown finish_reason with no tool_calls: content captured but error notice set."""
    mock_client = MagicMock()
    resp = make_completion(content="Some content.", finish_reason="function_call_deprecated")
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Question.")

    assert result.answer == "Some content."
    # An unexpected finish_reason that is not "stop"/"tool_calls"/None should be flagged
    assert result.error is not None
    assert "finish_reason" in result.error


def test_finish_reason_length_preserves_prior_tool_records(sample_df):
    """Tool records accumulated before a 'length' error are preserved in the result."""
    mock_client = MagicMock()
    tc = make_tool_call("call_1", "get_dataset_overview", {})
    resp1 = make_completion(tool_calls=[tc], finish_reason="tool_calls")
    resp2 = make_completion(content="Partial...", finish_reason="length")
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Overview then truncate.")

    assert result.error is not None
    assert "length" in result.error
    assert result.answer == ""
    assert len(result.tool_calls) == 1   # Round-1 record is still present
    assert result.tool_calls[0].tool_name == "get_dataset_overview"


# ── 9. Tool-Call Entry Validation ────────────────────────────────────────────

def test_validate_tool_call_entry_valid():
    """Structurally valid tool-call entry returns all fields without error."""
    tc = MagicMock()
    tc.id = "call_abc"
    tc.function.name = "get_dataset_overview"
    tc.function.arguments = "{}"

    tool_name, call_id, raw_args, err = _validate_tool_call_entry(tc)

    assert err is None
    assert tool_name == "get_dataset_overview"
    assert call_id == "call_abc"
    assert raw_args == "{}"


def test_validate_tool_call_entry_missing_id():
    """Missing call ID is detected and reported."""
    tc = MagicMock()
    tc.id = None
    tc.function.name = "get_dataset_overview"
    tc.function.arguments = "{}"

    tool_name, call_id, raw_args, err = _validate_tool_call_entry(tc)

    assert err is not None
    assert "id" in err.lower() or "missing" in err.lower()
    assert raw_args is None


def test_validate_tool_call_entry_missing_function():
    """Missing function attribute is detected and reported."""
    tc = MagicMock()
    tc.id = "call_x"
    tc.function = None

    tool_name, call_id, raw_args, err = _validate_tool_call_entry(tc)

    assert err is not None
    assert "function" in err.lower()
    assert raw_args is None


def test_validate_tool_call_entry_missing_function_name():
    """Empty/None function name is detected and reported."""
    tc = MagicMock()
    tc.id = "call_y"
    tc.function.name = None
    tc.function.arguments = "{}"

    tool_name, call_id, raw_args, err = _validate_tool_call_entry(tc)

    assert err is not None
    assert "name" in err.lower() or "missing" in err.lower()
    assert raw_args is None


def test_agent_malformed_tc_missing_id_breaks_loop(sample_df):
    """A tool-call entry with a missing ID causes a structured error, not a crash."""
    mock_client = MagicMock()

    bad_tc = MagicMock()
    bad_tc.id = None            # Invalid: no call ID
    bad_tc.function.name = "get_dataset_overview"
    bad_tc.function.arguments = "{}"

    resp = make_completion(tool_calls=[bad_tc], finish_reason="tool_calls")
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Malformed tool-call ID.")

    assert result.error is not None
    assert result.answer == ""
    assert result.rounds == 1


def test_agent_malformed_tc_missing_function_name_breaks_loop(sample_df):
    """A tool-call entry with a missing function name causes a structured error, not a crash."""
    mock_client = MagicMock()

    bad_tc = MagicMock()
    bad_tc.id = "call_z"
    bad_tc.function.name = None    # Invalid: no tool name
    bad_tc.function.arguments = "{}"

    resp = make_completion(tool_calls=[bad_tc], finish_reason="tool_calls")
    mock_client.chat.completions.create.return_value = resp

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(sample_df, "Malformed function name.")

    assert result.error is not None
    assert result.answer == ""
    assert result.rounds == 1


# ── 11. Bug 2 Regression Tests: Grouping & Aggregation Argument Mapping & Recovery ────

@pytest.fixture
def ground_truth_20x6_df():
    """Ground truth DataFrame: 20 rows, 6 columns matching user test scenario."""
    csv_file = Path(__file__).resolve().parent.parent / "ground_truth_20x6.csv"
    if csv_file.is_file():
        return pd.read_csv(csv_file)
    data = []
    # 10 employees in Alpha (mean salary 1550.0)
    for i in range(1, 11):
        data.append({
            "employee": f"E{i:02d}",
            "department": "Alpha",
            "experience_years": i,
            "salary": 1100.0 + (i - 1) * 100.0,
            "performance_score": 60 + i,
            "hours_per_week": 40,
        })
    # 10 employees in Beta (mean salary 2550.0)
    for i in range(11, 21):
        data.append({
            "employee": f"E{i:02d}",
            "department": "Beta",
            "experience_years": i,
            "salary": 2100.0 + (i - 11) * 100.0,
            "performance_score": 75 + (i - 10),
            "hours_per_week": 40,
        })
    return pd.DataFrame(data)


def test_group_and_aggregate_direct_intended_arguments(ground_truth_20x6_df):
    """Bug 2 regression: verify group_and_aggregate directly with intended arguments on 20x6 dataset."""
    res = group_and_aggregate(
        ground_truth_20x6_df,
        group_by=["department"],
        aggregations={
            "employee": ["count"],
            "salary": ["mean"],
        },
    )
    assert res["group_count"] == 2
    results = {row["department"]: row for row in res["results"]}

    assert "Alpha" in results
    assert results["Alpha"]["employee_count"] == 10
    assert results["Alpha"]["salary_mean"] == 1550.0

    assert "Beta" in results
    assert results["Beta"]["employee_count"] == 10
    assert results["Beta"]["salary_mean"] == 2550.0


def test_group_and_aggregate_preserves_invalid_column_validation(ground_truth_20x6_df):
    """Bug 2 regression: preserve strict validation against invalid column names like 'employee_count'."""
    # 1. Direct function raises ValueError
    with pytest.raises(ValueError, match="Aggregation column 'employee_count' does not exist"):
        group_and_aggregate(
            ground_truth_20x6_df,
            group_by=["department"],
            aggregations={"employee_count": ["count"], "salary": ["mean"]},
        )

    # 2. Dispatcher returns structured error
    model_out, raw_out = execute_tool(
        ground_truth_20x6_df,
        "group_and_aggregate",
        {"group_by": ["department"], "aggregations": {"employee_count": ["count"]}},
    )
    assert raw_out is None
    assert model_out["status"] == "error"
    assert "employee_count" in model_out["error"]
    assert "Available columns" in model_out["error"]


def test_agent_intended_group_and_aggregate_request(ground_truth_20x6_df):
    """Bug 2 regression: agent executes intended group_and_aggregate request successfully."""
    mock_client = MagicMock()

    tc = make_tool_call(
        "call_grp_1",
        "group_and_aggregate",
        {
            "group_by": ["department"],
            "aggregations": {
                "employee": ["count"],
                "salary": ["mean"],
            },
        },
    )
    resp1 = make_completion(tool_calls=[tc], finish_reason="tool_calls")
    resp2 = make_completion(
        content=(
            "Grouped employees by department:\n"
            "- Department Alpha has 10 employees with an average salary of 1550.0.\n"
            "- Department Beta has 10 employees with an average salary of 2550.0."
        ),
        finish_reason="stop",
    )
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(
        ground_truth_20x6_df,
        "Group employees by department and calculate employee count and average salary.",
    )

    assert result.execution_success is True
    assert result.error is None
    assert result.rounds == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "group_and_aggregate"
    assert result.tool_calls[0].output.get("group_count") == 2
    assert result.is_verified is True


def test_agent_recovers_from_invalid_aggregation_arguments(ground_truth_20x6_df):
    """Bug 2 regression: agent receives tool error on invalid column, retries with valid arguments, and succeeds."""
    mock_client = MagicMock()

    # Round 1: Model uses invalid aggregation column 'employee_count'
    bad_tc = make_tool_call(
        "call_bad_1",
        "group_and_aggregate",
        {
            "group_by": ["department"],
            "aggregations": {
                "employee_count": ["count"],
                "salary": ["mean"],
            },
        },
    )
    resp1 = make_completion(tool_calls=[bad_tc], finish_reason="tool_calls")

    # Round 2: After receiving the error, model corrects the arguments
    good_tc = make_tool_call(
        "call_good_2",
        "group_and_aggregate",
        {
            "group_by": ["department"],
            "aggregations": {
                "employee": ["count"],
                "salary": ["mean"],
            },
        },
    )
    resp2 = make_completion(tool_calls=[good_tc], finish_reason="tool_calls")

    # Round 3: Model returns final verified answer
    resp3 = make_completion(
        content=(
            "Grouped results by department:\n"
            "- Department Alpha has 10 employees with an average salary of 1550.0.\n"
            "- Department Beta has 10 employees with an average salary of 2550.0."
        ),
        finish_reason="stop",
    )
    mock_client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(
        ground_truth_20x6_df,
        "Group employees by department and calculate employee count and average salary.",
    )

    assert result.execution_success is True
    assert result.error is None
    assert result.rounds == 3
    assert len(result.tool_calls) == 2

    # Tool call 1 failed with error
    assert result.tool_calls[0].output["status"] == "error"
    assert "employee_count" in result.tool_calls[0].output["error"]

    # Tool call 2 succeeded
    assert result.tool_calls[1].output.get("group_count") == 2
    results = {row["department"]: row for row in result.tool_calls[1].output["results"]}
    assert results["Alpha"]["employee_count"] == 10
    assert results["Alpha"]["salary_mean"] == 1550.0
    assert results["Beta"]["employee_count"] == 10
    assert results["Beta"]["salary_mean"] == 2550.0

    # Grounding verified against the recovered successful tool call
    assert result.is_verified is True

    # Verify conversation history passed to mock_client in round 2 included the tool error message
    calls = mock_client.chat.completions.create.call_args_list
    round_2_messages = calls[1].kwargs["messages"]
    tool_msg = next(m for m in round_2_messages if m["role"] == "tool")
    tool_content = json.loads(tool_msg["content"])
    assert tool_content["status"] == "error"
    assert "employee_count" in tool_content["error"]


def test_tool_schema_and_system_prompt_instruct_aggregation_keys(ground_truth_20x6_df):
    """Bug 2 regression: verify schema and system prompt instruct model on aggregation keys and retry."""
    # Check group_and_aggregate tool schema description
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "group_and_aggregate")
    fn = schema["function"]
    agg_desc = fn["parameters"]["properties"]["aggregations"]["description"]
    assert "existing" in agg_desc.lower()
    assert "derived" in agg_desc.lower() or "source" in agg_desc.lower()

    # Check system prompt instructions
    agent = AnalysisAgent()
    sys_prompt = agent._create_system_prompt(ground_truth_20x6_df)
    assert "group_and_aggregate" in sys_prompt
    assert "existing" in sys_prompt.lower()
    assert "retry" in sys_prompt.lower() or "error" in sys_prompt.lower()


def test_tool_schema_and_system_prompt_instruct_correlation_pair_counts(ground_truth_20x6_df):
    """Bug A regression: verify schema and system prompt instruct model on reporting pair_counts."""
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "analyze_correlations")
    fn = schema["function"]
    assert "pair_counts" in fn["description"]

    agent = AnalysisAgent()
    sys_prompt = agent._create_system_prompt(ground_truth_20x6_df)
    assert "pair_counts" in sys_prompt
    assert "pairwise count" in sys_prompt.lower()


def test_agent_correlation_pair_counts_reported_on_generalization_dataset():
    """Bug A regression: verify agent runs analyze_correlations and receives pairwise counts 14, 14, 13."""
    import pathlib
    gen_path = pathlib.Path(__file__).resolve().parent.parent / "examples" / "generalization_test.csv"
    gen_df = pd.read_csv(gen_path)

    mock_client = MagicMock()
    tc = make_tool_call(
        "call_corr",
        "analyze_correlations",
        {"columns": ["years_experience", "monthly_salary", "satisfaction_score"]},
    )
    resp1 = make_completion(tool_calls=[tc])
    resp2 = make_completion(
        content=(
            "Analysis results:\n"
            "- years_experience and monthly_salary have a strong positive correlation of 0.99 (14 paired observations).\n"
            "- years_experience and satisfaction_score have a strong negative correlation of -0.97 (14 paired observations).\n"
            "- monthly_salary and satisfaction_score have a strong negative correlation of -0.98 (13 paired observations)."
        )
    )
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    agent = AnalysisAgent(client=mock_client)
    result = agent.run(
        gen_df,
        "Calculate correlations and paired observations between years_experience, monthly_salary, and satisfaction_score.",
    )

    assert result.execution_success is True
    assert result.rounds == 2
    assert len(result.tool_calls) == 1

    tool_out = result.tool_calls[0].output
    assert "pair_counts" in tool_out
    assert tool_out["pair_counts"]["years_experience"]["monthly_salary"] == 14
    assert tool_out["pair_counts"]["years_experience"]["satisfaction_score"] == 14
    assert tool_out["pair_counts"]["monthly_salary"]["satisfaction_score"] == 13

    # Grounding validator verifies without false flags
    assert result.is_flagged is False
    assert result.is_verified is True

