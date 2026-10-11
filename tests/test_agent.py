# tests/test_agent.py
"""
Tests for Phase 2, Step 1: Core LLM-powered data analysis agent.

All tests use a mocked OpenAI client and run completely offline without requiring
a real API key or making network requests.
"""

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
from app.agent.agent import build_dataset_metadata
from app.agent.tools import to_json_safe, REGISTERED_TOOL_NAMES


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


def make_completion(content: str = None, tool_calls: list = None):
    """Helper to create a mocked ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
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
