# tests/test_audit.py
"""
Comprehensive Testing and Audit Pass for the Applied ML Agent.

This audit module evaluates:
1. Tool-Use Discipline: registered tool constraints, argument validation, and dispatch isolation.
2. Hallucination Resistance: 14 deterministic scenarios examining model claims vs actual tool outputs.
3. Failure Transparency: 24 failure/boundary cases including finish_reason, error preservation,
   and ambiguous answer+error states.
4. Independent Analytical Validation: offline verification against examples/employee_analytics.csv
   calculated directly via pandas without calling the agent or tool dispatcher.

All tests run completely offline and deterministically with no external network or API calls.
"""

from unittest.mock import MagicMock
import json
import math
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from openai import OpenAIError, AuthenticationError, APITimeoutError, APIConnectionError

from app.agent import (
    AnalysisAgent,
    AgentResult,
    ToolCallRecord,
    TOOL_SCHEMAS,
    execute_tool,
)
from app.agent.tools import REGISTERED_TOOL_NAMES, to_json_safe
from app.agent.agent import (
    build_dataset_metadata,
    _parse_tool_arguments,
    _validate_tool_call_entry,
)
from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
    group_and_aggregate,
)


# ── Test Helpers ──────────────────────────────────────────────────────────────

def make_tc(call_id: str, name: str, arguments: dict | str = "{}"):
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


def make_resp(content: str = None, tool_calls: list = None, finish_reason: str = None):
    """Helper to create a mocked ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def employee_csv_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "examples" / "employee_analytics.csv"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: AUDIT TOOL-USE DISCIPLINE
# ══════════════════════════════════════════════════════════════════════════════

class TestToolUseDiscipline:
    """Verifies registered-tool restrictions, argument validation, and dispatch isolation."""

    def test_registered_tool_whitelist_exactness(self):
        """Only the six analytical tools are registered."""
        expected = {
            "get_dataset_overview",
            "get_summary_statistics",
            "filter_data",
            "analyze_correlations",
            "generate_visualization",
            "group_and_aggregate",
        }
        assert set(REGISTERED_TOOL_NAMES) == expected
        assert len(TOOL_SCHEMAS) == 6

    @pytest.mark.parametrize("dangerous_name", [
        "eval",
        "exec",
        "os.system",
        "subprocess.run",
        "__import__",
        "run_bash",
        "run_python",
        "read_file",
        "write_file",
    ])
    def test_reject_arbitrary_code_or_system_execution(self, clean_df, dangerous_name):
        """Dispatcher immediately rejects any attempt to call arbitrary execution tools."""
        model_out, raw_out = execute_tool(clean_df, dangerous_name, {"code": "import os; os.system('ls')"})
        assert model_out["status"] == "error"
        assert f"Unknown tool '{dangerous_name}'" in model_out["error"]
        assert raw_out is None

    def test_filter_data_rejects_nonexistent_column(self, clean_df):
        """filter_data rejects filtering on a column not in the DataFrame."""
        args = {"conditions": [{"column": "nonexistent_col", "operator": "eq", "value": 10}]}
        model_out, raw_out = execute_tool(clean_df, "filter_data", args)
        assert model_out["status"] == "error"
        assert "nonexistent_col" in model_out["error"]
        assert raw_out is None

    def test_filter_data_rejects_invalid_operator(self, clean_df):
        """filter_data rejects invalid filter operators."""
        args = {"conditions": [{"column": "x", "operator": "regex_match", "value": ".*"}]}
        model_out, raw_out = execute_tool(clean_df, "filter_data", args)
        assert model_out["status"] == "error"
        assert "Unknown operator" in model_out["error"] or "regex_match" in model_out["error"]
        assert raw_out is None

    def test_filter_data_rejects_missing_conditions_list(self, clean_df):
        """filter_data rejects empty conditions."""
        model_out, raw_out = execute_tool(clean_df, "filter_data", {"conditions": []})
        assert model_out["status"] == "error"
        assert "empty" in model_out["error"].lower()
        assert raw_out is None

    def test_filter_data_rejects_unexpected_nested_structure(self, clean_df):
        """filter_data rejects condition items that are not dicts."""
        args = {"conditions": ["not_a_dict"]}
        model_out, raw_out = execute_tool(clean_df, "filter_data", args)
        assert model_out["status"] == "error"
        assert raw_out is None

    def test_group_and_aggregate_rejects_nonexistent_group_column(self, clean_df):
        """group_and_aggregate rejects nonexistent grouping columns."""
        args = {"group_by": ["phantom_col"], "aggregations": {"x": ["mean"]}}
        model_out, raw_out = execute_tool(clean_df, "group_and_aggregate", args)
        assert model_out["status"] == "error"
        assert "phantom_col" in model_out["error"]
        assert raw_out is None

    def test_group_and_aggregate_rejects_unsupported_operation(self, clean_df):
        """group_and_aggregate rejects unsupported aggregation operations like 'mode' or 'std'."""
        args = {"group_by": ["category"], "aggregations": {"x": ["mode"]}}
        model_out, raw_out = execute_tool(clean_df, "group_and_aggregate", args)
        assert model_out["status"] == "error"
        assert "mode" in model_out["error"]
        assert raw_out is None

    def test_group_and_aggregate_rejects_numeric_operation_on_string_column(self, clean_df):
        """group_and_aggregate rejects numeric aggregations (mean, sum) on non-numeric columns."""
        args = {"group_by": ["x"], "aggregations": {"category": ["mean"]}}
        model_out, raw_out = execute_tool(clean_df, "group_and_aggregate", args)
        assert model_out["status"] == "error"
        assert "category" in model_out["error"]
        assert raw_out is None

    def test_group_and_aggregate_rejects_unexpected_nested_structure(self, clean_df):
        """group_and_aggregate rejects unexpected nested types in group_by."""
        args = {"group_by": [{"bad": "nested"}], "aggregations": {"x": ["mean"]}}
        model_out, raw_out = execute_tool(clean_df, "group_and_aggregate", args)
        assert model_out["status"] == "error"
        assert raw_out is None

    def test_generate_visualization_rejects_unknown_chart_type(self, clean_df):
        """generate_visualization rejects unsupported chart types."""
        args = {"chart_type": "pie", "x_column": "category"}
        model_out, raw_out = execute_tool(clean_df, "generate_visualization", args)
        assert model_out["status"] == "error"
        assert "pie" in model_out["error"]
        assert raw_out is None

    def test_generate_visualization_rejects_missing_required_axis(self, clean_df):
        """generate_visualization rejects scatter chart without y_column."""
        args = {"chart_type": "scatter", "x_column": "x"}
        model_out, raw_out = execute_tool(clean_df, "generate_visualization", args)
        assert model_out["status"] == "error"
        assert raw_out is None

    def test_analyze_correlations_rejects_nonexistent_column(self, clean_df):
        """analyze_correlations rejects nonexistent column in explicit columns list."""
        args = {"columns": ["x", "not_real"]}
        model_out, raw_out = execute_tool(clean_df, "analyze_correlations", args)
        assert model_out["status"] == "error"
        assert "not_real" in model_out["error"]
        assert raw_out is None

    def test_analyze_correlations_rejects_categorical_column_in_explicit_list(self, clean_df):
        """analyze_correlations rejects categorical columns in explicit columns list."""
        args = {"columns": ["x", "category"]}
        model_out, raw_out = execute_tool(clean_df, "analyze_correlations", args)
        assert model_out["status"] == "error"
        assert "category" in model_out["error"]
        assert raw_out is None

    def test_distinguishes_empty_filter_from_tool_error(self, clean_df):
        """A filter matching 0 rows is a valid result (status=success, matched_rows=0), NOT an error."""
        args = {"conditions": [{"column": "x", "operator": "gt", "value": 999}]}
        model_out, raw_out = execute_tool(clean_df, "filter_data", args)
        assert model_out["status"] == "success"
        assert model_out["matched_rows"] == 0
        assert model_out["preview"] == []
        assert isinstance(raw_out, pd.DataFrame)
        assert len(raw_out) == 0

    def test_dataframe_not_mutated_after_all_tools(self, clean_df):
        """DataFrame remains completely unmodified after running all analytical tools."""
        df_copy = clean_df.copy(deep=True)
        execute_tool(clean_df, "get_dataset_overview", {})
        execute_tool(clean_df, "get_summary_statistics", {"top_n": 2})
        execute_tool(clean_df, "filter_data", {"conditions": [{"column": "x", "operator": "gt", "value": 2}]})
        execute_tool(clean_df, "analyze_correlations", {})
        execute_tool(clean_df, "generate_visualization", {"chart_type": "scatter", "x_column": "x", "y_column": "y"})
        execute_tool(clean_df, "group_and_aggregate", {"group_by": ["category"], "aggregations": {"x": ["mean"]}})
        pd.testing.assert_frame_equal(clean_df, df_copy)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AUDIT HALLUCINATION RESISTANCE (14 SCENARIOS)
# ══════════════════════════════════════════════════════════════════════════════

class TestHallucinationResistance:
    """Deterministic tests examining application behavior when the model produces unsupported claims.

    CRITICAL ARCHITECTURAL FINDING:
    The application does NOT inspect the semantic content of the model's text response
    against tool outputs. Thus:
    - Hallucinations in final text answers are NOT detected by the application layer.
    - However, tool execution records in `AgentResult.tool_calls` and `AgentResult.figures`
      faithfully preserve objective execution ground truth.
    """

    def test_s01_numerical_contradiction(self, employee_df):
        """Scenario 1: Model numerical answer directly contradicts the actual tool result.

        Expected: Application executes tool correctly and records true output in tool_calls.
        Actual Behavior: Application records true tool output (mean = 2273750.0), but does
        not detect that the model's text claims '$50,000'. result.error is None.
        """
        mock_client = MagicMock()
        tc = make_tc(
            "call_agg",
            "group_and_aggregate",
            {"group_by": ["department"], "aggregations": {"salary": ["mean"]}},
        )
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="The average salary for Engineering is $50,000.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "What is the average salary in Engineering?")

        # 1. Tool result has ground truth
        assert len(result.tool_calls) == 1
        tool_results = result.tool_calls[0].output["results"]
        eng_rec = next(r for r in tool_results if r["department"] == "Engineering")
        assert eng_rec["salary_mean"] == 2273750.0

        # 2. Model text contradicts tool result
        assert "$50,000" in result.answer

        # 3. Application has NO claim-validation layer to catch this contradiction
        assert result.error is None

    def test_s02_category_invention(self, employee_df):
        """Scenario 2: Model invents a category ('Aerospace') not present in the dataset.

        Actual Behavior: Tool output lists true departments. Model text fabricates
        'Aerospace'. The application does not validate categories in answer against dataset.
        """
        mock_client = MagicMock()
        tc = make_tc("call_ov", "get_dataset_overview", {})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="The dataset includes an Aerospace department with 15 engineers.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "What departments exist?")

        assert len(result.tool_calls) == 1
        assert "Aerospace" in result.answer
        assert result.error is None  # Architectural limitation

    def test_s03_claims_tool_succeeded_when_no_tool_was_called(self, employee_df):
        """Scenario 3: Model claims that a tool succeeded when no tool was called.

        Actual Behavior: result.tool_calls is empty ([]), proving no tool ran.
        The caller can verify tool execution by checking len(result.tool_calls) == 0.
        """
        mock_client = MagicMock()
        resp = make_resp(
            content="I ran get_summary_statistics and calculated the average salary as $82,000.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Calculate the average salary.")

        # Ground truth: no tools were called
        assert len(result.tool_calls) == 0
        assert len(result.figures) == 0
        assert "ran get_summary_statistics" in result.answer
        assert result.error is None

    def test_s04_claims_tool_succeeded_when_tool_returned_error(self, employee_df):
        """Scenario 4: Model claims tool succeeded when the tool returned an error.

        Actual Behavior: Tool output contains status='error'. Model responds claiming
        success. Application preserves the error record in tool_calls, but does not flag
        the model's false success claim in result.error.
        """
        mock_client = MagicMock()
        # Invalid operator triggers tool error
        tc = make_tc(
            "call_flt",
            "filter_data",
            {"conditions": [{"column": "salary", "operator": "invalid_op", "value": 1000000}]},
        )
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="Filtering succeeded! Found 25 employees matching the criteria.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Filter employees.")

        # Ground truth: tool returned an error
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output["status"] == "error"
        assert "Unknown operator" in result.tool_calls[0].output["error"]
        # Model claims success despite error
        assert "Filtering succeeded" in result.answer
        assert result.error is None

    def test_s05_answer_despite_insufficient_evidence(self, employee_df):
        """Scenario 5: Model gives a speculative answer despite insufficient evidence.

        Actual Behavior: Model answers without analytical tools; tool_calls is empty.
        """
        mock_client = MagicMock()
        resp = make_resp(
            content="By 2030, salaries in this company will increase by 45% across all roles.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "What will salaries be in 2030?")

        assert len(result.tool_calls) == 0
        assert "2030" in result.answer
        assert result.error is None

    def test_s06_answers_using_only_dataset_preview(self, employee_df):
        """Scenario 6: Model answers using only the dataset preview when full calculation is required.

        Actual Behavior: Model computes from the 2-row preview in the system prompt
        without calling summary stats. len(result.tool_calls) == 0 exposes this.
        """
        mock_client = MagicMock()
        # Preview rows E001 (420000) and E002 (2400000) -> preview average is 1,410,000
        resp = make_resp(
            content="The average salary in the dataset is $1,410,000 based on the initial records.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "What is the average salary across the entire dataset?")

        # No tool was called — evidence of relying on prompt preview
        assert len(result.tool_calls) == 0
        assert "$1,410,000" in result.answer
        assert result.error is None

    def test_s07_treats_empty_filter_as_matching_rows(self, employee_df):
        """Scenario 7: Model treats an empty filter result as if matching rows existed.

        Actual Behavior: Tool result shows matched_rows=0. Model falsely claims 12 rows.
        Audit trail in tool_calls proves matched_rows == 0.
        """
        mock_client = MagicMock()
        tc = make_tc(
            "call_flt",
            "filter_data",
            {"conditions": [{"column": "salary", "operator": "gt", "value": 10000000}]},
        )
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="Found 12 employees earning over $10,000,000.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Find employees earning over 10M.")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output["matched_rows"] == 0
        assert "Found 12 employees" in result.answer
        assert result.error is None

    def test_s08_invents_correlation_coefficient(self, employee_df):
        """Scenario 8: Model invents a correlation coefficient without running the tool.

        Actual Behavior: No tool called; answer contains fabricated r value.
        """
        mock_client = MagicMock()
        resp = make_resp(
            content="The Pearson correlation between salary and experience is 0.99.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "What is the correlation between salary and experience?")

        assert len(result.tool_calls) == 0
        assert "0.99" in result.answer
        assert result.error is None

    def test_s09_claims_correlation_proves_causation(self, employee_df):
        """Scenario 9: Model claims that correlation proves causation.

        Actual Behavior: analyze_correlations computes r=0.8525. Model claims experience
        causes higher salary. Application has no causal verification check.
        """
        mock_client = MagicMock()
        tc = make_tc("call_corr", "analyze_correlations", {"columns": ["salary", "experience_years"]})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="The correlation is 0.85, which proves that gaining experience causes salary increases.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Does experience cause higher salary?")

        assert len(result.tool_calls) == 1
        assert "proves that gaining experience causes" in result.answer
        assert result.error is None

    def test_s10_claims_statistical_significance_without_test(self, employee_df):
        """Scenario 10: Model claims statistical significance even though no significance test exists.

        Actual Behavior: analyze_correlations provides no p-values. Model invents p < 0.001.
        """
        mock_client = MagicMock()
        tc = make_tc("call_corr", "analyze_correlations", {"columns": ["salary", "experience_years"]})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="The correlation of 0.85 is statistically significant at p < 0.001.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Is the correlation statistically significant?")

        assert len(result.tool_calls) == 1
        assert "p < 0.001" in result.answer
        assert result.error is None

    def test_s11_unsupported_claims_about_unavailable_columns(self, employee_df):
        """Scenario 11: Model makes claims about columns not available in the dataset.

        Actual Behavior: Model claims facts about a 'bonus' column not in employee_analytics.csv.
        """
        mock_client = MagicMock()
        resp = make_resp(
            content="The average bonus across employees is $15,000.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "What is the average bonus?")

        assert "bonus" not in employee_df.columns
        assert len(result.tool_calls) == 0
        assert "$15,000" in result.answer
        assert result.error is None

    def test_s12_ignores_contradictory_evidence_in_tool_results(self, clean_df):
        """Scenario 12: Model ignores contradictory evidence returned by analytical tools.

        Actual Behavior: Tool shows x and z have negative correlation (-1.0).
        Model asserts they have a strong positive correlation (+1.0).
        """
        mock_client = MagicMock()
        tc = make_tc("call_corr", "analyze_correlations", {"columns": ["x", "z"]})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="x and z have a strong positive correlation (+1.0).",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "How are x and z correlated?")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output["matrix"]["x"]["z"] == -1.0
        assert "positive correlation" in result.answer
        assert result.error is None

    def test_s13_confident_answer_to_ambiguous_question(self, employee_df):
        """Scenario 13: Model gives a confident answer to an ambiguous question without asking clarification.

        Actual Behavior: 'Top performer' could mean salary or performance_score.
        Model asserts one arbitrarily with full confidence.
        """
        mock_client = MagicMock()
        resp = make_resp(
            content="The undisputed top performer is E006 based entirely on experience.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Who is the top performer?")

        assert "E006" in result.answer
        assert result.error is None

    def test_s14_claims_visualization_generated_when_viz_failed(self, employee_df):
        """Scenario 14: Model claims to have generated a visualization when the tool failed.

        Actual Behavior: generate_visualization fails. Model claims the chart was created.
        Inspection of result.figures shows len(result.figures) == 0, and
        result.tool_calls[0].output['status'] == 'error'.
        """
        mock_client = MagicMock()
        tc = make_tc(
            "call_viz",
            "generate_visualization",
            {"chart_type": "pie", "x_column": "department"},  # 'pie' is unsupported
        )
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(
            content="I have generated a pie chart of departments for you.",
            finish_reason="stop",
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(employee_df, "Generate a pie chart of departments.")

        # Ground truth: no figures were generated and tool returned an error
        assert len(result.figures) == 0
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output["status"] == "error"
        assert "Unknown chart type" in result.tool_calls[0].output["error"]
        assert "pie chart" in result.answer
        assert result.error is None


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: AUDIT FAILURE TRANSPARENCY (24 CASES)
# ══════════════════════════════════════════════════════════════════════════════

class TestFailureTransparency:
    """Verifies all 24 failure/boundary cases and ensures error states are transparently recorded."""

    def test_case_01_missing_api_key(self, monkeypatch, clean_df):
        """Case 1: Missing API key raises clear ValueError before running."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = AnalysisAgent(api_key=None, client=None)
        with pytest.raises(ValueError, match="OpenAI API key is missing"):
            agent.run(clean_df, "Question")

    def test_case_02_invalid_api_key_authentication_error(self, clean_df):
        """Case 2: AuthenticationError from OpenAI produces structured error in result.error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Incorrect API key provided", response=MagicMock(status_code=401), body=None
        )
        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "OpenAI API error" in result.error
        assert result.answer == ""

    def test_case_03_generic_api_exception(self, clean_df):
        """Case 3: Generic OpenAI exception produces structured error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("Server error 500")
        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "OpenAI API error: Server error 500" in result.error
        assert result.answer == ""

    def test_case_04_network_or_timeout_failure(self, clean_df):
        """Case 4: Network timeout simulated without real network request."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "OpenAI API error" in result.error
        assert result.answer == ""

    def test_case_05_empty_response_choices(self, clean_df):
        """Case 5: Empty response choices list produces structured error."""
        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.choices = []
        mock_client.chat.completions.create.return_value = bad_resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "choices" in result.error.lower()
        assert result.answer == ""

    def test_case_06_malformed_response_structure(self, clean_df):
        """Case 6: Missing choices attribute produces structured error."""
        mock_client = MagicMock()
        bad_resp = MagicMock(spec=[])  # Object with no attributes
        mock_client.chat.completions.create.return_value = bad_resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "choices" in result.error.lower()
        assert result.answer == ""

    def test_case_07_missing_message_objects(self, clean_df):
        """Case 7: choice[0].message is None produces structured error."""
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message = None
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "message is None" in result.error
        assert result.answer == ""

    def test_case_08_missing_finish_reason_with_tool_calls(self, clean_df):
        """Case 8: Missing finish_reason (None) with tool_calls is routed to tool dispatch."""
        mock_client = MagicMock()
        tc = make_tc("call_1", "get_dataset_overview", {})
        resp1 = make_resp(tool_calls=[tc], finish_reason=None)
        resp2 = make_resp(content="Overview complete.", finish_reason=None)
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is None
        assert result.answer == "Overview complete."
        assert len(result.tool_calls) == 1

    def test_case_09_finish_reason_length(self, clean_df):
        """Case 9: finish_reason='length' produces structured error."""
        mock_client = MagicMock()
        resp = make_resp(content="Incomplete sentence...", finish_reason="length")
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "length" in result.error
        assert result.answer == ""

    def test_case_10_finish_reason_content_filter(self, clean_df):
        """Case 10: finish_reason='content_filter' produces structured error."""
        mock_client = MagicMock()
        resp = make_resp(content=None, finish_reason="content_filter")
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "content_filter" in result.error
        assert result.answer == ""

    def test_case_11_unknown_finish_reason_with_content_flags_error_and_preserves_content(self, clean_df):
        """Case 11: Unknown finish_reason with content sets BOTH answer and error.

        CRITICAL AUDIT ITEM:
        Demonstrates that AgentResult.answer contains text while AgentResult.error is also set.
        Callers checking only `bool(result.answer)` would treat this as complete success.
        Callers must check `result.error is None` to verify uncompromised success.
        """
        mock_client = MagicMock()
        resp = make_resp(content="Partial content from unusual provider.", finish_reason="provider_custom_stop")
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        # Both answer AND error are populated!
        assert result.answer == "Partial content from unusual provider."
        assert result.error is not None
        assert "provider_custom_stop" in result.error

        # Disambiguation: is_success is False because an error occurred
        assert result.is_success is False
        assert result.to_dict()["is_success"] is False

    def test_case_11b_is_success_property_semantics(self, clean_df):
        """Verify is_success cleanly distinguishes success from failure states."""
        # Clean success
        success_res = AgentResult(answer="Valid answer", error=None)
        assert success_res.is_success is True

        # Error with no answer
        err_res = AgentResult(answer="", error="Something failed")
        assert err_res.is_success is False

        # Error with partial answer
        mixed_res = AgentResult(answer="Partial answer", error="Truncated warning")
        assert mixed_res.is_success is False

        # Empty answer with no error
        empty_res = AgentResult(answer="   ", error=None)
        assert empty_res.is_success is False

    def test_case_12_malformed_tool_call_missing_id(self, clean_df):
        """Case 12 & 13: Malformed tool call missing ID breaks reasoning loop cleanly."""
        mock_client = MagicMock()
        bad_tc = MagicMock()
        bad_tc.id = None
        bad_tc.function.name = "get_dataset_overview"
        bad_tc.function.arguments = "{}"
        resp = make_resp(tool_calls=[bad_tc], finish_reason="tool_calls")
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "'id' is missing" in result.error
        assert result.answer == ""

    def test_case_14_malformed_tool_call_missing_function_name(self, clean_df):
        """Case 14: Malformed tool call missing function name breaks loop cleanly."""
        mock_client = MagicMock()
        bad_tc = MagicMock()
        bad_tc.id = "call_bad_fn"
        bad_tc.function.name = None
        bad_tc.function.arguments = "{}"
        resp = make_resp(tool_calls=[bad_tc], finish_reason="tool_calls")
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert "function name is missing" in result.error
        assert result.answer == ""

    def test_case_15_malformed_tool_arguments_sends_error_to_model(self, clean_df):
        """Case 15: Malformed JSON arguments returned as structured error to model for recovery."""
        mock_client = MagicMock()
        tc = make_tc("call_bad_arg", "filter_data", "NOT_JSON{{{")
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(content="Corrected after arg error.", finish_reason="stop")
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output["status"] == "error"
        assert "Invalid JSON" in result.tool_calls[0].output["error"]
        assert result.answer == "Corrected after arg error."

    def test_case_16_unknown_tool_name_handled_defensively(self, clean_df):
        """Case 16: Unknown tool name returns error dict to model, not unhandled exception."""
        mock_client = MagicMock()
        tc = make_tc("call_unk", "execute_sql_query", "{}")
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(content="SQL not supported.", finish_reason="stop")
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Run SQL query")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output["status"] == "error"
        assert "Unknown tool 'execute_sql_query'" in result.tool_calls[0].output["error"]
        assert result.answer == "SQL not supported."

    def test_case_17_and_18_tool_raises_exception_and_returns_structured_error(self, clean_df):
        """Cases 17 & 18: Tool function exception wrapped in status='error' structure."""
        mock_client = MagicMock()
        tc = make_tc("call_err", "filter_data", {"conditions": [{"column": "x", "operator": "bad_op", "value": 1}]})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(content="Handled tool error.", finish_reason="stop")
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Bad operator filter")

        assert result.tool_calls[0].output["status"] == "error"
        assert "Unknown operator" in result.tool_calls[0].output["error"]

    def test_case_19_max_rounds_exhaustion(self, clean_df):
        """Case 19: Infinite tool-calling loop terminated at max_rounds with clear error."""
        mock_client = MagicMock()
        tc = make_tc("call_loop", "get_dataset_overview", {})
        resp_loop = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        mock_client.chat.completions.create.return_value = resp_loop

        agent = AnalysisAgent(client=mock_client, max_rounds=2)
        result = agent.run(clean_df, "Loop forever")

        assert result.rounds == 2
        assert len(result.tool_calls) == 2
        assert result.error is not None
        assert "Exhausted maximum tool-calling rounds" in result.error
        assert result.answer == ""

    def test_case_20_successful_tool_followed_by_api_failure(self, clean_df):
        """Case 20 & 22: Prior successful tool call records preserved when later round fails."""
        mock_client = MagicMock()
        tc = make_tc("call_1", "get_dataset_overview", {})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        mock_client.chat.completions.create.side_effect = [resp1, OpenAIError("Connection reset")]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.error is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_dataset_overview"
        assert result.answer == ""

    def test_case_21_failure_after_figures_already_generated(self, clean_df):
        """Case 21: Figures generated in round 1 are preserved when round 2 fails."""
        mock_client = MagicMock()
        tc = make_tc("call_viz", "generate_visualization", {"chart_type": "scatter", "x_column": "x", "y_column": "y"})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        mock_client.chat.completions.create.side_effect = [resp1, OpenAIError("Timeout")]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Scatter plot then fail")

        assert result.error is not None
        assert len(result.figures) == 1
        assert isinstance(result.figures[0], go.Figure)
        assert result.answer == ""

    def test_case_23_unexpected_tool_output_structure_does_not_crash_agent(self, clean_df, monkeypatch):
        """Case 23: If execute_tool returns unexpected non-dict, JSON serialization handles it safely."""
        mock_client = MagicMock()
        tc = make_tc("call_cust", "get_dataset_overview", {})
        resp1 = make_resp(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_resp(content="Finished.", finish_reason="stop")
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Overview")

        assert result.error is None
        assert result.answer == "Finished."

    def test_case_24_non_finite_numeric_values_and_missing_values(self):
        """Case 24: Non-finite floats (NaN, Inf, -Inf) and pd.NA serialized to JSON nulls."""
        data = {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "pd_na": pd.NA,
            "np_nan": np.nan,
            "valid_float": 12.34567,
        }
        safe_data = to_json_safe(data)

        assert safe_data["nan"] is None
        assert safe_data["inf"] is None
        assert safe_data["neg_inf"] is None
        assert safe_data["pd_na"] is None
        assert safe_data["np_nan"] is None
        assert safe_data["valid_float"] == 12.3457

        # Must cleanly serialize to valid JSON
        serialized = json.dumps(safe_data)
        assert json.loads(serialized)["nan"] is None


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: INDEPENDENT ANALYTICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestIndependentAnalyticalValidation:
    """Validates analytical engine directly against examples/employee_analytics.csv.

    Expected values are independently calculated using pandas directly on the CSV file,
    completely separate from the agent and tool dispatcher.
    """

    @pytest.fixture(autouse=True)
    def load_data(self, employee_csv_path):
        """Load raw CSV directly with pandas for independent baseline calculation."""
        self.df = pd.read_csv(employee_csv_path)

    def test_01_dataset_dimensions(self):
        """1. Dataset dimensions independently calculated from CSV."""
        expected_shape = self.df.shape
        assert expected_shape == (50, 6)

        overview = get_dataset_overview(self.df)
        assert overview["rows"] == 50
        assert overview["columns"] == 6

    def test_02_column_names_and_data_types(self):
        """2. Column names and types match independent inspection."""
        expected_cols = ['employee_id', 'department', 'experience_years', 'salary', 'performance_score', 'remote']
        assert list(self.df.columns) == expected_cols

        overview = get_dataset_overview(self.df)
        assert overview["column_names"] == expected_cols
        assert "experience_years" in overview["dtypes"]
        assert "salary" in overview["dtypes"]

    def test_03_summary_statistics_numeric_columns(self):
        """3. Summary statistics match independent pandas computations."""
        stats = get_summary_statistics(self.df)
        numeric_stats = stats["numeric"]

        for col in ["experience_years", "salary", "performance_score"]:
            series = self.df[col].dropna()
            expected_mean = round(float(series.mean()), 4)
            expected_median = float(series.median())
            expected_std = round(float(series.std()), 4)
            expected_min = float(series.min())
            expected_max = float(series.max())

            actual = numeric_stats[col]
            assert actual["mean"] == pytest.approx(expected_mean, rel=1e-3)
            assert actual["median"] == pytest.approx(expected_median, rel=1e-3)
            assert actual["std"] == pytest.approx(expected_std, rel=1e-3)
            assert actual["min"] == pytest.approx(expected_min, rel=1e-3)
            assert actual["max"] == pytest.approx(expected_max, rel=1e-3)

    def test_04_average_salary_by_department(self):
        """4. Average salary by department matches independent pandas groupby."""
        expected_means = self.df.groupby("department")["salary"].mean().to_dict()

        agg_result = group_and_aggregate(
            self.df,
            group_by=["department"],
            aggregations={"salary": ["mean"]},
            dropna=True,
        )
        actual_results = {item["department"]: item["salary_mean"] for item in agg_result["results"]}

        for dept, exp_mean in expected_means.items():
            assert actual_results[dept] == pytest.approx(round(float(exp_mean), 4), rel=1e-3)

    def test_05_department_with_highest_average_salary(self):
        """5. Department with highest average salary independently verified."""
        dept_means = self.df.groupby("department")["salary"].mean()
        highest_dept = dept_means.idxmax()
        highest_val = dept_means.max()

        assert highest_dept == "Engineering"
        assert highest_val == pytest.approx(2273750.0)

        agg_result = group_and_aggregate(
            self.df,
            group_by=["department"],
            aggregations={"salary": ["mean"]},
        )
        sorted_results = sorted(agg_result["results"], key=lambda x: x["salary_mean"], reverse=True)
        top_dept = sorted_results[0]
        assert top_dept["department"] == "Engineering"
        assert top_dept["salary_mean"] == pytest.approx(2273750.0)

    def test_06_salary_and_experience_correlation(self):
        """6. Correlation between salary and experience matches independent calculation."""
        expected_corr = round(float(self.df[["salary", "experience_years"]].corr(method="pearson").iloc[0, 1]), 4)
        assert expected_corr == pytest.approx(0.8525, abs=1e-4)

        corr_res = analyze_correlations(self.df, columns=["salary", "experience_years"])
        actual_corr = corr_res["matrix"]["salary"]["experience_years"]
        assert actual_corr == pytest.approx(expected_corr, abs=1e-4)

    def test_07_filter_with_known_matching_row_count(self):
        """7. Filter salary > 2,000,000 matches independent row count."""
        expected_count = int((self.df["salary"] > 2000000).sum())
        assert expected_count == 14

        filtered = filter_data(
            self.df,
            conditions=[{"column": "salary", "operator": "gt", "value": 2000000}],
        )
        assert len(filtered) == 14

    def test_08_empty_filter_result(self):
        """8. Filter salary > 10,000,000 matches independent count of 0."""
        expected_count = int((self.df["salary"] > 10000000).sum())
        assert expected_count == 0

        filtered = filter_data(
            self.df,
            conditions=[{"column": "salary", "operator": "gt", "value": 10000000}],
        )
        assert len(filtered) == 0

    def test_09_missing_value_handling(self):
        """9. Missing values match independent null counts."""
        expected_missing = self.df.isnull().sum().to_dict()
        assert expected_missing["department"] == 2
        assert expected_missing["experience_years"] == 1
        assert expected_missing["salary"] == 4
        assert expected_missing["performance_score"] == 5
        assert expected_missing["remote"] == 3

        overview = get_dataset_overview(self.df)
        assert overview["missing_counts"]["department"] == 2
        assert overview["missing_counts"]["experience_years"] == 1
        assert overview["missing_counts"]["salary"] == 4
        assert overview["missing_counts"]["performance_score"] == 5
        assert overview["missing_counts"]["remote"] == 3

    def test_10_numerical_edge_case_constant_column(self):
        """10. Constant column correlation returns None and warning."""
        edge_df = pd.DataFrame({"constant": [5.0, 5.0, 5.0], "varying": [1.0, 2.0, 3.0]})
        res = analyze_correlations(edge_df)
        assert res["matrix"]["constant"]["varying"] is None
        assert res["warning"] is not None

    def test_11_grouped_aggregation_multiple_grouping_columns(self):
        """11. Multi-column grouping ['department', 'remote'] matches independent pandas result."""
        expected_df = (
            self.df.groupby(["department", "remote"], dropna=True)["salary"]
            .agg(["mean", "count"])
            .reset_index()
        )
        expected_count = len(expected_df)
        assert expected_count == 20

        agg_result = group_and_aggregate(
            self.df,
            group_by=["department", "remote"],
            aggregations={"salary": ["mean", "count"]},
            dropna=True,
        )
        assert len(agg_result["results"]) == 20

        # Check specific known group: Data Science, Yes
        ds_yes_exp = expected_df[(expected_df["department"] == "Data Science") & (expected_df["remote"] == "Yes")].iloc[0]
        ds_yes_act = next(
            r for r in agg_result["results"]
            if r["department"] == "Data Science" and r["remote"] == "Yes"
        )
        assert ds_yes_act["salary_count"] == int(ds_yes_exp["count"])
        assert ds_yes_act["salary_mean"] == pytest.approx(round(float(ds_yes_exp["mean"]), 4), rel=1e-3)

    def test_12_aggregation_with_missing_values_dropna_false(self):
        """12. dropna=False includes missing department groups."""
        expected_groups_dropna_true = self.df["department"].dropna().nunique()
        assert expected_groups_dropna_true == 11

        res_dropna_true = group_and_aggregate(
            self.df,
            group_by=["department"],
            aggregations={"salary": ["mean", "count"]},
            dropna=True,
        )
        assert len(res_dropna_true["results"]) == 11

        res_dropna_false = group_and_aggregate(
            self.df,
            group_by=["department"],
            aggregations={"salary": ["mean", "count"]},
            dropna=False,
        )
        assert len(res_dropna_false["results"]) == 12

        # Check the missing department group
        missing_dept_rec = next(r for r in res_dropna_false["results"] if r["department"] is None)
        assert missing_dept_rec["salary_count"] == 2
        assert missing_dept_rec["salary_mean"] == pytest.approx(1725000.0)

    def test_json_serialization_preserves_values(self):
        """Clean JSON serialization preserves precision and semantic meaning."""
        agg_result = group_and_aggregate(
            self.df,
            group_by=["department"],
            aggregations={"salary": ["mean"]},
        )
        serialized = json.dumps(to_json_safe(agg_result))
        deserialized = json.loads(serialized)
        assert len(deserialized["results"]) == 11

    def test_input_dataframe_remains_unchanged(self):
        """Original employee DataFrame remains unmutated across analytical calls."""
        copy_df = self.df.copy(deep=True)
        get_dataset_overview(self.df)
        get_summary_statistics(self.df)
        filter_data(self.df, conditions=[{"column": "salary", "operator": "gt", "value": 1000000}])
        analyze_correlations(self.df)
        group_and_aggregate(self.df, group_by=["department"], aggregations={"salary": ["mean"]})
        pd.testing.assert_frame_equal(self.df, copy_df)
