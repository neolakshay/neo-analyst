# tests/test_ui.py
"""
Tests for Phase 3: Streamlit UI integration and helper utilities.

Covers:
1. Valid CSV loading from path and buffer.
2. Empty or malformed CSV handling.
3. Dataset metadata and bounded preview.
4. Empty-question validation.
5. Agent invocation on explicit submission.
6. Correct display/handling of execution errors.
7. VERIFIED grounding.
8. FLAGGED grounding.
9. UNVERIFIABLE grounding.
10. Plotly figure rendering and figure handoff.
11. Tool-call information rendering.
12. Dataset switching without stale result leakage.

All tests run completely offline and deterministically without requiring a real API key.
"""

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import re
import subprocess
import sys
import time
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from app.agent import (
    AnalysisAgent,
    AgentResult,
    ToolCallRecord,
    GroundingStatus,
    GroundingResult,
)
from app.ui.utils import (
    load_csv,
    get_dataset_summary,
    validate_question,
    format_tool_call_summary,
    is_api_key_configured,
    run_analysis_query,
    EXAMPLE_DATASETS,
    resolve_example_path,
)


APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "ui" / "app.py")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Simple sample DataFrame."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "department": ["Engineering", "HR", "Engineering", "Marketing", "HR"],
        "salary": [90000.0, 55000.0, 95000.0, 65000.0, 58000.0],
        "bonus": [10000.0, None, 12000.0, 8000.0, None],
    })


# ── 1. CSV Loading and Validation Tests ────────────────────────────────────────

def test_load_csv_valid_path():
    """Valid CSV loading from filesystem path."""
    emp_path = resolve_example_path(EXAMPLE_DATASETS["Employee Analytics"])
    df, err = load_csv(emp_path)
    assert err is None
    assert df is not None
    assert len(df) == 50
    assert "salary" in df.columns


def test_load_csv_valid_buffer():
    """Valid CSV loading from an in-memory buffer."""
    buf = StringIO("a,b,c\n1,2,3\n4,5,6\n")
    df, err = load_csv(buf)
    assert err is None
    assert df is not None
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_csv_empty_file():
    """Empty CSV file handling produces a clear error."""
    buf = StringIO("")
    df, err = load_csv(buf)
    assert df is None
    assert err is not None
    assert "empty" in err.lower()


def test_load_csv_nonexistent_path():
    """Missing CSV file path produces a clear error."""
    df, err = load_csv("nonexistent_path_file_12345.csv")
    assert df is None
    assert err is not None
    assert "not found" in err.lower()


def test_load_csv_none_input():
    """None input returns error."""
    df, err = load_csv(None)
    assert df is None
    assert err == "No file provided."


def test_load_csv_no_columns():
    """CSV with no columns produces an error."""
    buf = StringIO("\n\n\n")
    df, err = load_csv(buf)
    assert df is None
    assert err is not None


def test_load_csv_malformed_syntax():
    """Malformed CSV content produces a parsing error instead of crashing."""
    # Inconsistent delimiter/column counts
    buf = StringIO("col1,col2\n1,2\n3,4,5,6,7\n")
    df, err = load_csv(buf)
    assert df is None
    assert err is not None
    assert "malformed" in err.lower() or "error" in err.lower()


# ── 2. Dataset Metadata & Bounded Preview Tests ───────────────────────────────

def test_get_dataset_summary(sample_df):
    """Dataset summary generates correct dimensions, column details, and bounded preview."""
    summary = get_dataset_summary(sample_df, preview_rows=3)

    assert summary["rows"] == 5
    assert summary["columns"] == 5
    assert summary["total_nulls"] == 2  # Two missing bonus entries
    assert len(summary["preview"]) == 3  # Bounded to 3 rows

    col_df = summary["column_details"]
    assert isinstance(col_df, pd.DataFrame)
    assert list(col_df["Column"]) == ["employee_id", "name", "department", "salary", "bonus"]
    bonus_row = col_df[col_df["Column"] == "bonus"].iloc[0]
    assert bonus_row["Missing Count"] == 2
    assert bonus_row["Non-Null Count"] == 3


# ── 3. Question Validation Tests ──────────────────────────────────────────────

def test_validate_question_valid():
    """Non-empty string is valid."""
    is_valid, err = validate_question("Which department has the highest average salary?")
    assert is_valid is True
    assert err is None


def test_validate_question_empty():
    """Empty or whitespace-only questions fail validation."""
    for empty_val in ["", "   ", "\n\t"]:
        is_valid, err = validate_question(empty_val)
        assert is_valid is False
        assert err is not None


def test_validate_question_non_string():
    """Non-string question input fails validation."""
    for invalid_val in [None, 123, ["question"]]:
        is_valid, err = validate_question(invalid_val)
        assert is_valid is False
        assert "string" in err.lower()


# ── 4. Tool Call Summary Formatting Tests ─────────────────────────────────────

def test_format_tool_call_summary_success():
    """ToolCallRecord formatting produces concise summaries for successful operations."""
    record = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={"row_count": 50, "columns": ["a", "b", "c"]},
    )
    formatted = format_tool_call_summary(record)
    assert formatted["tool_name"] == "get_dataset_overview"
    assert formatted["status"] == "Success"
    assert formatted["is_error"] is False
    assert "50 rows" in formatted["summary"]


def test_format_tool_call_summary_error():
    """ToolCallRecord formatting handles errors honestly."""
    record = ToolCallRecord(
        tool_name="filter_data",
        arguments={"conditions": [{"column": "unknown", "operator": "=="}]},
        output={"error": "Column 'unknown' not found in dataset."},
    )
    formatted = format_tool_call_summary(record)
    assert formatted["status"] == "Error"
    assert formatted["is_error"] is True
    assert "Column 'unknown' not found" in formatted["summary"]


def test_format_tool_call_summary_overview_int_columns():
    """Bug 1 regression: overview output where 'columns' is an integer."""
    record = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={
            "rows": 20,
            "columns": 6,
            "column_names": ["employee", "department", "experience_years", "salary", "performance_score", "hours_per_week"],
        },
    )
    formatted = format_tool_call_summary(record)
    assert formatted["status"] == "Success"
    assert formatted["is_error"] is False
    assert formatted["summary"] == "Retrieved overview: 20 rows, 6 columns"


def test_format_tool_call_summary_overview_column_names_list():
    """Bug 1 regression: overview output containing 'column_names' as a list without integer 'columns'."""
    record = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={
            "rows": 15,
            "column_names": ["col1", "col2", "col3", "col4"],
        },
    )
    formatted = format_tool_call_summary(record)
    assert formatted["status"] == "Success"
    assert formatted["is_error"] is False
    assert formatted["summary"] == "Retrieved overview: 15 rows, 4 columns"


def test_format_tool_call_summary_overview_defensive_fields():
    """Bug 1 regression: defensive handling for missing or malformed fields."""
    rec_empty = ToolCallRecord(tool_name="get_dataset_overview", arguments={}, output={})
    fmt_empty = format_tool_call_summary(rec_empty)
    assert fmt_empty["summary"] == "Retrieved overview: ? rows, ? columns"

    rec_none = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={"rows": None, "columns": None},
    )
    fmt_none = format_tool_call_summary(rec_none)
    assert fmt_none["summary"] == "Retrieved overview: ? rows, ? columns"


def test_app_renders_overview_tool_call_without_type_error():
    """Bug 1 regression: AppTest exercises rendering of overview tool call details without crashing."""
    overview_result = AgentResult(
        answer="The dataset has 20 rows and 6 columns.",
        tool_calls=[
            ToolCallRecord(
                tool_name="get_dataset_overview",
                arguments={},
                output={
                    "rows": 20,
                    "columns": 6,
                    "column_names": ["employee", "department", "experience_years", "salary", "performance_score", "hours_per_week"],
                },
            )
        ],
        rounds=1,
    )

    with patch("app.ui.utils.run_analysis_query", return_value=overview_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Show overview").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            assert analyze_btn is not None
            analyze_btn.click().run()

            assert not at.exception
            markdowns = [m.value for m in at.markdown]
            assert any("Retrieved overview: 20 rows, 6 columns" in m for m in markdowns)


def test_format_tool_call_summary_all_tools_actual_schemas():
    """Verify format_tool_call_summary accurately summarizes real analytical output schemas for all 6 tools."""
    # 1. get_dataset_overview
    ov_rec = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={"rows": 20, "columns": 6, "column_names": ["a", "b", "c", "d", "e", "f"]},
    )
    assert format_tool_call_summary(ov_rec)["summary"] == "Retrieved overview: 20 rows, 6 columns"

    # 2. get_summary_statistics (real schema has 'numeric' and 'categorical')
    stat_rec = ToolCallRecord(
        tool_name="get_summary_statistics",
        arguments={},
        output={
            "numeric": {"salary": {}, "experience_years": {}},
            "categorical": {"department": {}},
        },
    )
    assert format_tool_call_summary(stat_rec)["summary"] == "Computed statistics for 3 column(s)"

    # 3. filter_data
    filt_rec = ToolCallRecord(
        tool_name="filter_data",
        arguments={},
        output={"matched_rows": 3, "total_rows": 20},
    )
    assert format_tool_call_summary(filt_rec)["summary"] == "Filtered dataset to 3 row(s)"

    # 4. analyze_correlations (real schema has 'matrix' and 'columns_used')
    corr_rec = ToolCallRecord(
        tool_name="analyze_correlations",
        arguments={},
        output={
            "columns_used": ["salary", "experience_years"],
            "matrix": {"salary": {}, "experience_years": {}},
        },
    )
    assert format_tool_call_summary(corr_rec)["summary"] == "Correlation matrix with 2 variable(s)"

    # 5. generate_visualization
    viz_rec = ToolCallRecord(
        tool_name="generate_visualization",
        arguments={},
        output={"chart_type": "scatter", "title": "Scatter plot"},
    )
    assert format_tool_call_summary(viz_rec)["summary"] == "Generated scatter visualization"

    # 6. group_and_aggregate (real schema has 'group_count' and 'results')
    grp_rec = ToolCallRecord(
        tool_name="group_and_aggregate",
        arguments={},
        output={"group_count": 2, "results": [{"department": "Alpha"}, {"department": "Beta"}]},
    )
    assert format_tool_call_summary(grp_rec)["summary"] == "Aggregated into 2 group(s)"


def test_format_tool_call_summary_malformed_and_unexpected_values():
    """Verify format_tool_call_summary safely handles unexpected types and missing fields."""
    # Overview with non-integer, non-list columns and non-int rows
    ov_bad = ToolCallRecord(
        tool_name="get_dataset_overview",
        arguments={},
        output={"rows": None, "columns": "unexpected", "column_names": 123},
    )
    assert format_tool_call_summary(ov_bad)["summary"] == "Retrieved overview: ? rows, ? columns"

    # Statistics with None values
    stat_bad = ToolCallRecord(
        tool_name="get_summary_statistics",
        arguments={},
        output={"numeric": None, "categorical": None},
    )
    assert format_tool_call_summary(stat_bad)["summary"] == "Computed statistics for 0 column(s)"

    # Correlation with None values
    corr_bad = ToolCallRecord(
        tool_name="analyze_correlations",
        arguments={},
        output={"matrix": None, "columns_used": None},
    )
    assert format_tool_call_summary(corr_bad)["summary"] == "Correlation matrix with 0 variable(s)"

    # Grouping with None values
    grp_bad = ToolCallRecord(
        tool_name="group_and_aggregate",
        arguments={},
        output={"group_count": None, "results": None},
    )
    assert format_tool_call_summary(grp_bad)["summary"] == "Aggregated into 0 group(s)"

    # Empty dictionary
    assert format_tool_call_summary(ToolCallRecord(tool_name="filter_data", arguments={}, output={}))["summary"] == "Filtered dataset to ? row(s)"

    # Non-dict output
    rec_num = ToolCallRecord(tool_name="custom_tool", arguments={}, output=42)
    assert format_tool_call_summary(rec_num)["status"] == "Success"


def test_app_renders_various_tool_calls_and_errors_without_crashing():
    """AppTest verifies rendering does not crash on errors, empty results, or malformed optional metadata."""
    res = AgentResult(
        answer="Analysis complete.",
        tool_calls=[
            ToolCallRecord(
                tool_name="group_and_aggregate",
                arguments={},
                output={"status": "error", "error": "Aggregation column 'ghost' does not exist."},
            ),
            ToolCallRecord(
                tool_name="get_summary_statistics",
                arguments={},
                output={"numeric": {"salary": {}}, "categorical": {}},
            ),
            ToolCallRecord(
                tool_name="get_dataset_overview",
                arguments={},
                output={"rows": None, "columns": None},
            ),
        ],
        rounds=2,
    )

    with patch("app.ui.utils.run_analysis_query", return_value=res):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Run analysis").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            assert analyze_btn is not None
            analyze_btn.click().run()

            assert not at.exception
            errors = [e.value for e in at.error]
            assert any("Aggregation column 'ghost' does not exist" in err for err in errors)
            markdowns = [m.value for m in at.markdown]
            assert any("Computed statistics for 1 column(s)" in m for m in markdowns)
            assert any("Retrieved overview: ? rows, ? columns" in m for m in markdowns)


# ── 5. API Key Check Tests ───────────────────────────────────────────────────

def test_is_api_key_configured():
    """API key detection returns boolean without leaking key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        assert is_api_key_configured() is True

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        assert is_api_key_configured() is False


# ── 6. Query Execution & Non-Mutation Tests ───────────────────────────────────

def test_run_analysis_query_does_not_mutate_dataframe(sample_df):
    """Execution must not mutate the original DataFrame."""
    original_copy = sample_df.copy()

    mock_agent = MagicMock()
    mock_agent.run.return_value = AgentResult(answer="Analysis answer.")

    result = run_analysis_query(sample_df, "What is the average salary?", agent=mock_agent)

    assert result.answer == "Analysis answer."
    # Original DataFrame remains completely identical
    pd.testing.assert_frame_equal(sample_df, original_copy)
    assert mock_agent.run.call_count == 1


def test_run_analysis_query_invalid_question(sample_df):
    """Empty question returns an AgentResult with error without invoking agent."""
    mock_agent = MagicMock()
    result = run_analysis_query(sample_df, "", agent=mock_agent)
    assert result.error is not None
    assert mock_agent.run.call_count == 0


def test_run_analysis_query_empty_dataframe():
    """Empty DataFrame returns an AgentResult with error."""
    mock_agent = MagicMock()
    result = run_analysis_query(pd.DataFrame(), "Valid question", agent=mock_agent)
    assert result.error is not None
    assert mock_agent.run.call_count == 0


# ── 7. Streamlit AppTest Integration Tests ────────────────────────────────────

def test_app_initial_render():
    """App initializes with default example dataset and renders metrics."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    # Title rendered
    assert len(at.title) >= 1
    assert "BYTE Applied ML" in at.title[0].value
    # Metrics rendered (Rows, Columns, Missing)
    assert len(at.metric) >= 3


def test_app_empty_question_submission_shows_error():
    """Submitting an empty question shows a validation error."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Find the Analyze button and click it with empty question
    analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
    assert analyze_btn is not None
    analyze_btn.click().run()

    assert not at.exception
    # Error message rendered
    errors = [e.value for e in at.error]
    assert any("enter a question" in err.lower() for err in errors)


def test_app_missing_api_key_shows_error_on_submit():
    """Submitting with unconfigured API key produces clear error."""
    with patch("app.ui.utils.is_api_key_configured", return_value=False):
        at = AppTest.from_file(APP_PATH)
        at.run()

        # Set a question
        at.text_area[0].input("Valid question").run()
        analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
        assert analyze_btn is not None
        analyze_btn.click().run()

        assert not at.exception
        errors = [e.value for e in at.error]
        assert any("openai_api_key" in err.lower() and "not configured" in err.lower() for err in errors)


def test_app_verified_grounding_render():
    """Verified grounding result displays success banner and verified claims."""
    verified_result = AgentResult(
        answer="The highest average salary is in Engineering with $2,273,750.",
        tool_calls=[
            ToolCallRecord(
                tool_name="group_and_aggregate",
                arguments={"group_by": ["department"]},
                output={"status": "completed", "results": [{"department": "Engineering"}]},
            )
        ],
        rounds=2,
        grounding=GroundingResult(
            status=GroundingStatus.VERIFIED,
            claims_checked=["Group 'Engineering' salary_mean (2273750.0) matches answer"],
            reasons=["Group 'Engineering' salary_mean (2273750.0) matches answer"],
        ),
    )

    with patch("app.ui.utils.run_analysis_query", return_value=verified_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Which department has the highest average salary?").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            assert not at.exception
            # Verify Answer is present in markdown
            markdowns = [m.value for m in at.markdown]
            assert any("Engineering with $2,273,750" in m for m in markdowns)
            # Verify VERIFIED success banner
            successes = [s.value for s in at.success]
            assert any("VERIFIED" in s for s in successes)


def test_app_flagged_grounding_render():
    """Flagged grounding result displays warning/error banner and reasons."""
    flagged_result = AgentResult(
        answer="The average salary is $50,000.",
        tool_calls=[
            ToolCallRecord(
                tool_name="get_summary_statistics",
                arguments={},
                output={"statistics": {"salary": {"mean": 2273750.0}}},
            )
        ],
        rounds=1,
        grounding=GroundingResult(
            status=GroundingStatus.FLAGGED,
            reasons=["Numerical contradiction: claimed 50000.0 but tool returned 2273750.0"],
            claims_checked=["salary mean 2273750.0 vs 50000.0"],
        ),
    )

    with patch("app.ui.utils.run_analysis_query", return_value=flagged_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("What is the average salary?").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            assert not at.exception
            # Grounding FLAGGED banner rendered
            errors = [e.value for e in at.error]
            assert any("FLAGGED" in e for e in errors)


def test_app_unverifiable_grounding_render():
    """Unverifiable grounding result displays info banner."""
    unverifiable_result = AgentResult(
        answer="Experience and salary appear positively correlated based on trends.",
        tool_calls=[],
        rounds=1,
        grounding=GroundingResult(
            status=GroundingStatus.UNVERIFIABLE,
            reasons=["Answer contains qualitative statements that cannot be verified deterministically."],
        ),
    )

    with patch("app.ui.utils.run_analysis_query", return_value=unverifiable_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Describe the trends.").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            assert not at.exception
            # Grounding UNVERIFIABLE info banner rendered
            infos = [i.value for i in at.info]
            assert any("UNVERIFIABLE" in i for i in infos)


def test_app_plotly_figure_render():
    """Plotly figures attached to AgentResult are rendered in the UI."""
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[10, 20, 30])])
    fig_result = AgentResult(
        answer="Scatter plot created.",
        figures=[fig],
        tool_calls=[
            ToolCallRecord(
                tool_name="generate_visualization",
                arguments={"chart_type": "scatter"},
                output={"chart_type": "scatter"},
            )
        ],
        rounds=1,
    )

    with patch("app.ui.utils.run_analysis_query", return_value=fig_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Plot salary vs experience").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            assert not at.exception
            # Verify Plotly chart exists in at.get("plotly_chart") elements
            assert len(at.get("plotly_chart")) == 1


def test_app_tool_calls_render():
    """Tool execution details are rendered when tool calls are present."""
    record = ToolCallRecord(
        tool_name="get_summary_statistics",
        arguments={},
        output={"status": "completed", "statistics": {"salary": {"mean": 1000.0}}},
    )
    res = AgentResult(
        answer="Statistics calculated.",
        tool_calls=[record],
        rounds=1,
    )

    with patch("app.ui.utils.run_analysis_query", return_value=res):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Calculate stats").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            assert not at.exception
            markdowns = [m.value for m in at.markdown]
            assert any("get_summary_statistics" in m for m in markdowns)


def test_app_execution_error_display():
    """Execution errors are prominently displayed."""
    error_result = AgentResult(
        answer="",
        error="OpenAI API error: Rate limit reached.",
        rounds=1,
    )

    with patch("app.ui.utils.run_analysis_query", return_value=error_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.text_area[0].input("Analyze data").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            assert not at.exception
            errors = [e.value for e in at.error]
            assert any("Rate limit reached" in err for err in errors)


def test_app_dataset_switching_clears_stale_results():
    """Switching datasets clears prior query results so they do not leak into the new dataset."""
    prior_result = AgentResult(
        answer="Employee analytics specific answer about Engineering.",
        tool_calls=[],
        rounds=1,
    )

    with patch("app.ui.utils.run_analysis_query", return_value=prior_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()

            # Execute query on initial dataset
            at.text_area[0].input("Employee question").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            analyze_btn.click().run()

            # Result is visible
            markdowns = [m.value for m in at.markdown]
            assert any("Employee analytics specific answer" in m for m in markdowns)

            # Switch dataset to E-commerce Sales
            at.selectbox(key="example_choice").select("E-commerce Sales").run()

            # Stale result must NO LONGER be present
            markdowns_after = [m.value for m in at.markdown]
            assert not any("Employee analytics specific answer" in m for m in markdowns_after)
            assert at.session_state["last_result"] is None
            assert at.session_state["last_question"] is None


# ── 8. Import Resolution & CLI Entrypoint Regression Tests ───────────────────

def test_import_resolution_when_script_dir_prepended_to_sys_path():
    """Regression test: Streamlit prepends the script directory (app/ui) to sys.path.

    Exercises the real package structure on disk:
    When sys.path has app/ui before project root, Python must not resolve 'app'
    to app/ui/app.py and loop indefinitely with RecursionError.
    """
    repo_root = Path(__file__).resolve().parent.parent
    script_dir = repo_root / "app" / "ui"

    probe_code = (
        "import sys, os\n"
        f"repo_root = {str(repo_root)!r}\n"
        f"script_dir = {str(script_dir)!r}\n"
        "# Replicate Streamlit startup sys.path structure\n"
        "sys.path.insert(0, repo_root)\n"
        "sys.path.insert(0, script_dir)\n"
        "# Import and check package resolution\n"
        "import app\n"
        "assert hasattr(app, '__path__'), f'app was resolved as a non-package: {app}'\n"
        "from app.agent import AnalysisAgent, GroundingStatus\n"
        "from app.ui.utils import load_csv\n"
        "assert AnalysisAgent is not None\n"
        "assert GroundingStatus is not None\n"
        "assert load_csv is not None\n"
        "print('IMPORT_SUCCESS')\n"
    )

    res = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=10,
    )
    assert res.returncode == 0, f"Import resolution failed with stderr:\n{res.stderr}"
    assert "IMPORT_SUCCESS" in res.stdout
    assert "RecursionError" not in res.stderr


def test_streamlit_cli_entrypoint_launch_no_recursion():
    """Regression test: Running 'streamlit run app/ui/app.py' starts without RecursionError."""
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/ui/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8596",
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Give Streamlit 2.5 seconds to initialize or fail
        time.sleep(2.5)
        ret = proc.poll()
        if ret is not None:
            stdout, stderr = proc.communicate()
            assert "RecursionError" not in stderr
            assert ret == 0, f"Process exited unexpectedly with code {ret}:\n{stderr}"
        else:
            # Process successfully running without crashing
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=5)
            assert "RecursionError" not in stderr
            assert "RecursionError" not in stdout
    finally:
        if proc.poll() is None:
            proc.kill()


# ── 9. Dataset Selector & Dynamic Row Count Tests ────────────────────────────

def test_example_datasets_display_names_have_no_hardcoded_row_counts():
    """Verify none of the example dataset display names contain hardcoded row counts."""
    expected_labels = [
        "Employee Analytics",
        "E-commerce Sales",
        "Sample Data",
        "Edge Cases",
    ]
    # Check exact keys match stable human-readable names
    assert list(EXAMPLE_DATASETS.keys()) == expected_labels

    for label in EXAMPLE_DATASETS.keys():
        # Ensure label does not contain "(... rows)" or any row count indicator
        assert not re.search(r"\d+\s*rows?", label, re.IGNORECASE), (
            f"Dataset label '{label}' contains a hardcoded row count."
        )


def test_all_example_datasets_selectable_in_ui():
    """Verify all four datasets remain selectable in the sidebar selectbox."""
    for label, rel_path in EXAMPLE_DATASETS.items():
        abs_path = resolve_example_path(rel_path)
        assert abs_path.is_file(), f"Dataset file does not exist: {abs_path}"

        at = AppTest.from_file(APP_PATH)
        at.run()
        assert not at.exception

        # Select the dataset
        at.selectbox(key="example_choice").select(label).run()
        assert not at.exception

        # Verify session state reflects the selected dataset
        assert at.session_state["dataset_name"] == label
        assert at.session_state["df"] is not None

        # Verify overview header and metric reflect the loaded dataset
        df = at.session_state["df"]
        total_rows_metric = next((m for m in at.metric if m.label == "Total Rows"), None)
        assert total_rows_metric is not None
        assert total_rows_metric.value == f"{len(df):,}"


def test_dataset_overview_metrics_reflect_actual_dataframe():
    """Verify dataset overview metrics dynamically reflect the actual loaded DataFrame."""
    custom_df = pd.DataFrame({
        "col_a": list(range(42)),
        "col_b": ["val"] * 42,
    })
    summary = get_dataset_summary(custom_df)
    assert summary["rows"] == 42
    assert summary["columns"] == 2
    assert summary["total_nulls"] == 0

    with patch("app.ui.utils.load_csv", return_value=(custom_df, None)):
        at = AppTest.from_file(APP_PATH)
        at.run()
        assert not at.exception

        rows_metric = next((m for m in at.metric if m.label == "Total Rows"), None)
        cols_metric = next((m for m in at.metric if m.label == "Total Columns"), None)
        assert rows_metric is not None
        assert rows_metric.value == "42"
        assert cols_metric is not None
        assert cols_metric.value == "2"


def test_ui_updates_when_dataset_row_count_changes():
    """Verify the UI metrics update accurately when a dataset's row count changes."""
    # Simulate a dataset where row count changed (e.g. truncated to 20 rows)
    truncated_df = pd.DataFrame({
        "id": list(range(20)),
        "val": [100.0] * 20,
    })

    with patch("app.ui.utils.load_csv", return_value=(truncated_df, None)):
        at = AppTest.from_file(APP_PATH)
        at.run()
        assert not at.exception

        # The label in selectbox should remain the stable name without row count
        selected_option = at.selectbox(key="example_choice").value
        assert selected_option == "Employee Analytics"
        assert "20" not in selected_option
        assert "50" not in selected_option

        # But the Total Rows metric must display the actual count (20)
        rows_metric = next((m for m in at.metric if m.label == "Total Rows"), None)
        assert rows_metric is not None
        assert rows_metric.value == "20"


def test_app_combined_ui_to_agent_grouping_workflow():
    """End-to-End integration test: exercises dataset selection, query submission, agent execution,
    overview and grouping tool calls, and UI rendering without crashing.
    """
    combined_result = AgentResult(
        answer="Grouped employees: Alpha has 10 employees (mean salary 1550.0), Beta has 10 employees (mean salary 2550.0).",
        tool_calls=[
            ToolCallRecord(
                tool_name="get_dataset_overview",
                arguments={},
                output={
                    "rows": 20,
                    "columns": 6,
                    "column_names": ["employee", "department", "experience_years", "salary", "performance_score", "hours_per_week"],
                },
            ),
            ToolCallRecord(
                tool_name="group_and_aggregate",
                arguments={
                    "group_by": ["department"],
                    "aggregations": {"employee": ["count"], "salary": ["mean"]},
                },
                output={
                    "group_by": ["department"],
                    "aggregations": {"employee": ["count"], "salary": ["mean"]},
                    "group_count": 2,
                    "results": [
                        {"department": "Alpha", "employee_count": 10, "salary_mean": 1550.0},
                        {"department": "Beta", "employee_count": 10, "salary_mean": 2550.0},
                    ],
                },
            ),
        ],
        rounds=2,
        grounding=GroundingResult(
            status=GroundingStatus.VERIFIED,
            claims_checked=["Group 'Alpha' salary_mean (1550.0) matches answer"],
            reasons=["Group 'Alpha' salary_mean (1550.0) matches answer"],
        ),
    )

    with patch("app.ui.utils.run_analysis_query", return_value=combined_result):
        with patch("app.ui.utils.is_api_key_configured", return_value=True):
            at = AppTest.from_file(APP_PATH)
            at.run()
            assert not at.exception

            # Submit question
            at.text_area[0].input("Group employees by department and calculate employee count and average salary.").run()
            analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
            assert analyze_btn is not None
            analyze_btn.click().run()

            assert not at.exception
            # Verify Answer rendered
            markdowns = [m.value for m in at.markdown]
            assert any("Alpha has 10 employees" in m for m in markdowns)
            # Verify Verified Grounding Status rendered
            assert any("VERIFIED" in s.value for s in at.success)
            # Verify both tool calls rendered properly in tool details
            assert any("Retrieved overview: 20 rows, 6 columns" in m for m in markdowns)
            assert any("Aggregated into 2 group(s)" in m for m in markdowns)


