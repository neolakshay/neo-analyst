# app/ui/utils.py
"""
app.ui.utils
------------
Helper utilities for dataset loading, metadata extraction, tool call
formatting, and agent execution in the Streamlit user interface.
"""

from pathlib import Path
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from dotenv import load_dotenv

from app.agent import AnalysisAgent, AgentResult, ToolCallRecord

# Ensure local .env is loaded
load_dotenv()

EXAMPLE_DATASETS: Dict[str, str] = {
    "Employee Analytics": "examples/employee_analytics.csv",
    "E-commerce Sales": "examples/ecommerce_sales.csv",
    "Sample Data": "examples/sample_data.csv",
    "Edge Cases": "examples/edge_cases.csv",
}


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_example_path(relative_path: str) -> Path:
    """Resolve an example dataset relative path to an absolute Path."""
    return get_project_root() / relative_path


def load_csv(
    file_or_path: Union[str, Path, io.BytesIO, io.StringIO, Any],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Safely load a CSV file or file-like object into a pandas DataFrame.

    Parameters
    ----------
    file_or_path : Union[str, Path, io.BytesIO, io.StringIO, Any]
        A filesystem path, Path object, or file-like buffer (e.g., Streamlit UploadedFile).

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        (df, None) on success, or (None, error_message) on failure.
    """
    if file_or_path is None:
        return None, "No file provided."

    # If it's a file path string or Path object, verify existence
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        if not p.is_file():
            return None, f"Dataset file not found at: {p}"
        source = p
    else:
        # File-like object (UploadedFile, BytesIO, StringIO)
        if hasattr(file_or_path, "seek"):
            try:
                file_or_path.seek(0)
            except Exception:
                pass
        source = file_or_path

    try:
        try:
            df = pd.read_csv(source)
        except UnicodeDecodeError:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(source, encoding="latin-1")
    except pd.errors.EmptyDataError:
        return None, "The CSV file is empty or contains no data."
    except pd.errors.ParserError as parse_err:
        return None, f"Failed to parse CSV. The file appears malformed: {str(parse_err)}"
    except Exception as exc:
        return None, f"Failed to read CSV: {str(exc)}"

    if not isinstance(df, pd.DataFrame):
        return None, "Loaded object is not a valid DataFrame."

    if len(df.columns) == 0:
        return None, "The CSV file contains no columns."

    return df, None


def get_dataset_summary(df: pd.DataFrame, preview_rows: int = 10) -> Dict[str, Any]:
    """Extract bounded overview metadata and tabular preview for the UI.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    preview_rows : int, default 10
        Maximum number of rows to return in the preview.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing rows count, columns count, column details DataFrame,
        total missing cells, and bounded preview DataFrame.
    """
    n_rows, n_cols = df.shape

    col_records: List[Dict[str, Any]] = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        non_null_count = n_rows - null_count
        col_records.append({
            "Column": str(col),
            "Data Type": str(df[col].dtype),
            "Non-Null Count": non_null_count,
            "Missing Count": null_count,
        })

    col_df = pd.DataFrame(col_records)
    total_nulls = int(df.isnull().sum().sum())
    preview_df = df.head(preview_rows)

    return {
        "rows": n_rows,
        "columns": n_cols,
        "column_details": col_df,
        "total_nulls": total_nulls,
        "preview": preview_df,
    }


def validate_question(question: Any) -> Tuple[bool, Optional[str]]:
    """Validate a natural-language question string.

    Parameters
    ----------
    question : Any
        The input question to validate.

    Returns
    -------
    Tuple[bool, Optional[str]]
        (True, None) if valid, or (False, error_message) if invalid.
    """
    if not isinstance(question, str):
        return False, "Question must be a string."
    clean = question.strip()
    if not clean:
        return False, "Please enter a question to analyze."
    return True, None


def format_tool_call_summary(tool_record: ToolCallRecord) -> Dict[str, Any]:
    """Format a ToolCallRecord for concise UI display.

    Parameters
    ----------
    tool_record : ToolCallRecord
        The record of an executed tool call.

    Returns
    -------
    Dict[str, Any]
        Formatted dictionary with tool_name, arguments, status, summary, and raw_output.
    """
    name = tool_record.tool_name
    args = tool_record.arguments
    output = tool_record.output

    is_error = False
    summary_text = "Executed successfully."

    if isinstance(output, dict):
        if "error" in output:
            is_error = True
            summary_text = f"Error: {output['error']}"
        elif name == "get_dataset_overview":
            rows = output.get("rows")
            if rows is None:
                rows = output.get("row_count", "?")

            cols_val = output.get("columns")
            if isinstance(cols_val, int):
                cols = cols_val
            elif isinstance(cols_val, (list, tuple, set)):
                cols = len(cols_val)
            elif "column_names" in output and isinstance(output["column_names"], (list, tuple, set)):
                cols = len(output["column_names"])
            else:
                cols = "?"

            summary_text = f"Retrieved overview: {rows} rows, {cols} columns"
        elif name == "get_summary_statistics":
            if isinstance(output.get("numeric"), dict) or isinstance(output.get("categorical"), dict):
                n_num = len(output["numeric"]) if isinstance(output.get("numeric"), dict) else 0
                n_cat = len(output["categorical"]) if isinstance(output.get("categorical"), dict) else 0
                cols_analyzed = n_num + n_cat
            elif isinstance(output.get("numeric_columns"), (list, dict, set, tuple)):
                cols_analyzed = len(output["numeric_columns"])
            elif isinstance(output.get("statistics"), (list, dict, set, tuple)):
                cols_analyzed = len(output["statistics"])
            else:
                cols_analyzed = 0
            summary_text = f"Computed statistics for {cols_analyzed} column(s)"
        elif name == "filter_data":
            matched = output.get("matched_rows", "?")
            summary_text = f"Filtered dataset to {matched} row(s)"
        elif name == "analyze_correlations":
            mat = output.get("matrix")
            if isinstance(mat, dict):
                n_vars = len(mat)
            elif isinstance(output.get("columns_used"), (list, tuple, set)):
                n_vars = len(output["columns_used"])
            elif isinstance(output.get("correlation_matrix"), dict):
                n_vars = len(output["correlation_matrix"])
            else:
                n_vars = 0
            summary_text = f"Correlation matrix with {n_vars} variable(s)"
        elif name == "generate_visualization":
            chart_type = output.get("chart_type", "chart")
            summary_text = f"Generated {chart_type} visualization"
        elif name == "group_and_aggregate":
            if isinstance(output.get("group_count"), int):
                groups = output["group_count"]
            elif isinstance(output.get("groups_count"), int):
                groups = output["groups_count"]
            elif isinstance(output.get("results"), (list, tuple)):
                groups = len(output["results"])
            else:
                groups = 0
            summary_text = f"Aggregated into {groups} group(s)"
        else:
            summary_text = output.get("message", "Tool execution completed.")
    elif isinstance(output, str):
        if "error" in output.lower():
            is_error = True
        summary_text = output[:200]
    elif output is None:
        summary_text = "No output returned."

    return {
        "tool_name": name,
        "arguments": args,
        "is_error": is_error,
        "status": "Error" if is_error else "Success",
        "summary": summary_text,
        "output": output,
    }


def is_api_key_configured() -> bool:
    """Check whether OPENAI_API_KEY is configured without exposing its value."""
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return bool(key)


def run_analysis_query(
    df: pd.DataFrame,
    question: str,
    agent: Optional[AnalysisAgent] = None,
) -> AgentResult:
    """Execute an analytical query against a dataset using AnalysisAgent.

    Ensures the original DataFrame is not mutated by passing a defensive copy.

    Parameters
    ----------
    df : pd.DataFrame
        The tabular dataset to analyze.
    question : str
        The natural-language question.
    agent : Optional[AnalysisAgent]
        Existing or mocked agent instance. If omitted, a new AnalysisAgent is created.

    Returns
    -------
    AgentResult
        The result of the agent execution.
    """
    valid, err = validate_question(question)
    if not valid:
        return AgentResult(
            answer="",
            error=err,
        )

    if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        return AgentResult(
            answer="",
            error="A valid DataFrame with at least one column is required.",
        )

    if agent is None:
        agent = AnalysisAgent()

    # Pass defensive copy to guarantee original df is not mutated
    df_copy = df.copy()
    return agent.run(df_copy, question.strip())
