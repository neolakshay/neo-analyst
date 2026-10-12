# app/agent/tools.py
"""
app.agent.tools
---------------
Tool definitions (OpenAI tool calling schemas) and local execution dispatcher
for the six analytical functions:
  1. get_dataset_overview
  2. get_summary_statistics
  3. filter_data
  4. analyze_correlations
  5. generate_visualization
  6. group_and_aggregate
"""

from typing import Any, Tuple
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
    group_and_aggregate,
)


def to_json_safe(val: Any) -> Any:
    """Recursively convert NumPy scalars, NaNs, and infinities into JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, dict):
        return {str(k): to_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [to_json_safe(item) for item in val]
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if pd.isna(val):
        return None
    return str(val)


# ── OpenAI Tool Calling Schemas ───────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": (
                "Get a structural overview of the dataset: number of rows, columns, "
                "column names, data types, missing value counts/percentages, and duplicate row count."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary_statistics",
            "description": (
                "Compute descriptive statistics for all columns: mean, median, std, min, max, "
                "q25, q75 for numeric columns, and top value frequencies for categorical columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top frequent values to return for categorical columns (default 5).",
                        "default": 5,
                        "minimum": 1,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_data",
            "description": (
                "Filter rows of the dataset using one or more conditions combined with AND. "
                "Supported operators: 'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'contains', 'isnull', 'notnull'. "
                "Returns the matched row count and a bounded preview of matching records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "description": "List of filter condition objects.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {
                                    "type": "string",
                                    "description": "The name of the column to filter on.",
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains", "isnull", "notnull"],
                                    "description": "The comparison operator.",
                                },
                                "value": {
                                    "description": "The target value for comparison (not required for isnull or notnull).",
                                },
                            },
                            "required": ["column", "operator"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    }
                },
                "required": ["conditions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_correlations",
            "description": (
                "Compute pairwise Pearson correlation coefficients between numeric columns. "
                "Returns the correlation matrix ('matrix'), pairwise complete observation counts ('pair_counts'), "
                "and the top positive (r > 0) and top negative (r < 0) pairs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of numeric column names to correlate. If omitted, all numeric columns are used.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of strongest correlation pairs to highlight (default 5).",
                        "default": 5,
                        "minimum": 1,
                    },
                    "min_periods": {
                        "type": "integer",
                        "description": "Minimum valid observations required per pair (default 1).",
                        "default": 1,
                        "minimum": 1,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": (
                "Generate an interactive Plotly chart. The chart figure is preserved for application use, "
                "and a confirmation summary is returned to the model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "histogram", "box"],
                        "description": "The type of chart to generate.",
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column for x-axis (required for bar, line, scatter; optional distribution column for histogram or grouping for box).",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for y-axis (required for bar, line, scatter; numeric column for box).",
                    },
                    "color_column": {
                        "type": "string",
                        "description": "Optional column name to use for color encoding.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional custom title for the chart.",
                    },
                },
                "required": ["chart_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "group_and_aggregate",
            "description": (
                "Group data by one or more columns and compute summary statistics. "
                "Supported operations: 'mean', 'median', 'sum', 'min', 'max', 'count'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of existing column names from the dataset to group by.",
                        "minItems": 1,
                    },
                    "aggregations": {
                        "type": "object",
                        "description": (
                            "Mapping of existing dataset column names to lists of aggregation operations, "
                            "e.g. {'salary': ['mean'], 'employee': ['count']}. "
                            "Every key MUST be an existing source column name in the dataset (NOT a derived name like 'employee_count'). "
                            "The operations in the list specify what to calculate on that source column."
                        ),
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["mean", "median", "sum", "min", "max", "count"],
                            },
                            "minItems": 1,
                        },
                    },
                    "dropna": {
                        "type": "boolean",
                        "description": "Whether to exclude rows with missing values in group_by columns (default True).",
                        "default": True,
                    },
                },
                "required": ["group_by", "aggregations"],
                "additionalProperties": False,
            },
        },
    },
]

REGISTERED_TOOL_NAMES = {tool["function"]["name"] for tool in TOOL_SCHEMAS}


# ── Tool Dispatcher ───────────────────────────────────────────────────────────

def execute_tool(
    df: pd.DataFrame,
    tool_name: str,
    arguments: dict,
    preview_rows: int = 5,
) -> Tuple[dict, Any]:
    """Execute a registered analytical tool locally and return clean results.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    tool_name : str
        Name of the tool to execute.
    arguments : dict
        Parsed keyword arguments for the tool.
    preview_rows : int, default 5
        Maximum number of sample rows to include in bounded DataFrame previews.

    Returns
    -------
    Tuple[dict, Any]
        (model_output, raw_output) where:
        - model_output is a JSON-serializable dictionary formatted for LLM consumption.
        - raw_output is the original analytical result (dict, pd.DataFrame, or go.Figure).
    """
    if tool_name not in REGISTERED_TOOL_NAMES:
        return (
            {
                "status": "error",
                "error": f"Unknown tool '{tool_name}'. Available tools: {sorted(REGISTERED_TOOL_NAMES)}",
            },
            None,
        )

    try:
        if tool_name == "get_dataset_overview":
            res = get_dataset_overview(df)
            return to_json_safe(res), res

        elif tool_name == "get_summary_statistics":
            top_n = arguments.get("top_n", 5)
            res = get_summary_statistics(df, top_n=top_n)
            return to_json_safe(res), res

        elif tool_name == "filter_data":
            conditions = arguments.get("conditions", [])
            filtered_df = filter_data(df, conditions=conditions)
            preview_subset = filtered_df.head(preview_rows)
            preview_records = preview_subset.to_dict(orient="records")
            model_out = {
                "status": "success",
                "matched_rows": int(len(filtered_df)),
                "total_rows": int(len(df)),
                "preview_rows": int(len(preview_subset)),
                "columns": list(filtered_df.columns),
                "preview": to_json_safe(preview_records),
            }
            return model_out, filtered_df

        elif tool_name == "analyze_correlations":
            columns = arguments.get("columns")
            top_n = arguments.get("top_n", 5)
            min_periods = arguments.get("min_periods", 1)
            res = analyze_correlations(
                df, columns=columns, top_n=top_n, min_periods=min_periods
            )
            return to_json_safe(res), res

        elif tool_name == "generate_visualization":
            chart_type = arguments.get("chart_type", "")
            x_column = arguments.get("x_column")
            y_column = arguments.get("y_column")
            color_column = arguments.get("color_column")
            title = arguments.get("title")
            fig = generate_visualization(
                df,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                title=title,
            )
            chart_title = title or (fig.layout.title.text if fig.layout.title else f"{chart_type} chart")
            model_out = {
                "status": "success",
                "message": f"Visualization '{chart_type}' generated successfully.",
                "chart_type": chart_type,
                "title": chart_title,
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column,
            }
            return model_out, fig

        elif tool_name == "group_and_aggregate":
            group_by = arguments.get("group_by", [])
            aggregations = arguments.get("aggregations", {})
            dropna = arguments.get("dropna", True)
            res = group_and_aggregate(
                df, group_by=group_by, aggregations=aggregations, dropna=dropna
            )
            return to_json_safe(res), res

    except Exception as exc:
        err_msg = str(exc)
        if tool_name == "group_and_aggregate" and "Aggregation column" in err_msg and "does not exist" in err_msg:
            err_msg += (
                ". Note: aggregation keys must be existing column names from the dataset "
                "(aggregation operations specify what to calculate, e.g. use an existing column with 'count'). "
                "Please retry with valid column names."
            )
        return {"status": "error", "error": err_msg}, None

    return {"status": "error", "error": f"Unhandled tool '{tool_name}'"}, None
