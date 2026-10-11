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
    emp_path = resolve_example_path(EXAMPLE_DATASETS["Employee Analytics (50 rows)"])
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
            at.selectbox(key="example_choice").select("E-commerce Sales (100 rows)").run()

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

