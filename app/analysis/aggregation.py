# app/analysis/aggregation.py
"""
group_and_aggregate
-------------------
Groups tabular data by one or more columns and computes summary aggregations
for each group.

Supported aggregation operations
--------------------------------
  mean   – arithmetic average of non-null values (numeric columns only)
  median – middle value of non-null values (numeric columns only)
  sum    – total sum of non-null values (numeric columns only)
  min    – minimum non-null value
  max    – maximum non-null value
  count  – count of non-null observations in the target column

The function returns a plain Python dictionary suitable for JSON serialization.
"""

from typing import Any
import numpy as np
import pandas as pd

# The set of supported aggregation operations.
VALID_AGGREGATIONS = {"mean", "median", "sum", "min", "max", "count"}

# Operations that require numeric data.
NUMERIC_AGGREGATIONS = {"mean", "median", "sum"}


def _clean_group_val(val: Any) -> Any:
    """Convert grouping values to standard Python types, mapping NaN/None to None."""
    if pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isinf(val) or np.isnan(val):
            return None
        return float(val)
    return str(val)


def _clean_agg_val(val: Any, op: str) -> Any:
    """Convert aggregated values to clean, JSON-serializable Python types."""
    if pd.isna(val):
        return None
    if op == "count":
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isinf(val) or np.isnan(val):
            return None
        return round(float(val), 4)
    return str(val)


def group_and_aggregate(
    df: pd.DataFrame,
    group_by: list[str],
    aggregations: dict[str, list[str]],
    dropna: bool = True,
) -> dict:
    """Group rows of *df* and calculate summary statistics for each group.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to group and aggregate.
    group_by : list[str]
        One or more column names defining the grouping keys.
    aggregations : dict[str, list[str]]
        Mapping of column names to lists of aggregation operations.
        Supported operations: "mean", "median", "sum", "min", "max", "count".
    dropna : bool, default True
        Whether to exclude rows with missing values in the grouping columns.
        If False, missing values are treated as distinct group keys.

    Returns
    -------
    dict with keys:
        "group_by"    – list of grouping column names
        "aggregations"– normalized mapping of columns to aggregation operations
        "group_count" – number of distinct groups returned
        "results"     – list of dicts, each representing one group with its key(s)
                        and aggregated values named as "{column}_{operation}"

    Raises
    ------
    TypeError
        If *df* is not a DataFrame, *group_by* is not a list of strings,
        *aggregations* is not a dict of lists of strings, or *dropna* is not a bool.
    ValueError
        If *df* has no columns, *group_by* is empty or contains unknown/duplicate columns,
        *aggregations* is empty or references unknown columns/unsupported operations,
        numeric operations are applied to non-numeric columns, or an output name collides.
    """
    # ── 1. Validate df ────────────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if len(df.columns) == 0:
        raise ValueError("The DataFrame has no columns — nothing to group.")

    # ── 2. Validate dropna ────────────────────────────────────────────────────
    if not isinstance(dropna, bool):
        raise TypeError(
            f"'dropna' must be a boolean, got {type(dropna).__name__}."
        )

    # ── 3. Validate group_by ──────────────────────────────────────────────────
    if not isinstance(group_by, list):
        raise TypeError(
            f"'group_by' must be a list of column names, got {type(group_by).__name__}."
        )

    if len(group_by) == 0:
        raise ValueError("'group_by' must contain at least one column name.")

    seen_group_cols: set[str] = set()
    for col in group_by:
        if not isinstance(col, str):
            raise TypeError(
                f"All column names in 'group_by' must be strings, got {type(col).__name__}."
            )
        if col not in df.columns:
            raise ValueError(
                f"Grouping column '{col}' does not exist. "
                f"Available columns: {list(df.columns)}"
            )
        if col in seen_group_cols:
            raise ValueError(
                f"'group_by' contains duplicate column: '{col}'."
            )
        seen_group_cols.add(col)

    # ── 4. Validate aggregations ──────────────────────────────────────────────
    if not isinstance(aggregations, dict):
        raise TypeError(
            f"'aggregations' must be a dictionary, got {type(aggregations).__name__}."
        )

    if len(aggregations) == 0:
        raise ValueError("'aggregations' dictionary must contain at least one column mapping.")

    normalized_aggregations: dict[str, list[str]] = {}
    output_names: set[str] = set(group_by)

    for col, ops in aggregations.items():
        if not isinstance(col, str):
            raise TypeError(
                f"All column names in 'aggregations' must be strings, got {type(col).__name__}."
            )
        if col not in df.columns:
            raise ValueError(
                f"Aggregation column '{col}' does not exist. "
                f"Available columns: {list(df.columns)}"
            )
        if not isinstance(ops, list):
            raise TypeError(
                f"Operations for column '{col}' must be a list of strings, got {type(ops).__name__}."
            )
        if len(ops) == 0:
            raise ValueError(
                f"Operations list for column '{col}' cannot be empty."
            )

        col_is_numeric = pd.api.types.is_numeric_dtype(df[col])
        norm_ops: list[str] = []
        seen_ops: set[str] = set()

        for op in ops:
            if not isinstance(op, str):
                raise TypeError(
                    f"All operations for column '{col}' must be strings, got {type(op).__name__}."
                )
            norm_op = op.strip().lower()
            if norm_op not in VALID_AGGREGATIONS:
                raise ValueError(
                    f"Unsupported aggregation operation '{op}' for column '{col}'. "
                    f"Supported operations: {sorted(VALID_AGGREGATIONS)}"
                )
            if not df.empty and norm_op in NUMERIC_AGGREGATIONS and not col_is_numeric:
                raise ValueError(
                    f"Operation '{norm_op}' requires a numeric column, "
                    f"but column '{col}' has dtype '{df[col].dtype}'."
                )
            if norm_op in seen_ops:
                raise ValueError(
                    f"Duplicate aggregation operation '{op}' specified for column '{col}'."
                )
            seen_ops.add(norm_op)
            norm_ops.append(norm_op)

            out_col_name = f"{col}_{norm_op}"
            if out_col_name in output_names:
                raise ValueError(
                    f"Output column name collision: '{out_col_name}' collides with "
                    "a grouping column or another aggregated output column."
                )
            output_names.add(out_col_name)

        normalized_aggregations[col] = norm_ops

    # ── 5. Handle empty DataFrame edge case ───────────────────────────────────
    if df.empty:
        return {
            "group_by": list(group_by),
            "aggregations": normalized_aggregations,
            "group_count": 0,
            "results": [],
        }

    # ── 6. Group and aggregate ────────────────────────────────────────────────
    # observed=True keeps only observed categories and prevents deprecation warnings
    grouped = df.groupby(group_by, dropna=dropna, as_index=True, observed=True)
    agg_res = grouped.agg(normalized_aggregations)

    if agg_res.empty:
        return {
            "group_by": list(group_by),
            "aggregations": normalized_aggregations,
            "group_count": 0,
            "results": [],
        }

    # ── 7. Build structured, JSON-serializable records ─────────────────────────
    results: list[dict[str, Any]] = []

    for idx, row in agg_res.iterrows():
        record: dict[str, Any] = {}

        # Unpack grouping keys (single-column -> scalar, multi-column -> tuple)
        group_keys = [idx] if len(group_by) == 1 else list(idx)
        for g_col, g_val in zip(group_by, group_keys):
            record[g_col] = _clean_group_val(g_val)

        # Unpack aggregated columns
        for col, op_list in normalized_aggregations.items():
            for op in op_list:
                out_name = f"{col}_{op}"
                raw_val = row[(col, op)]
                record[out_name] = _clean_agg_val(raw_val, op)

        results.append(record)

    return {
        "group_by": list(group_by),
        "aggregations": normalized_aggregations,
        "group_count": len(results),
        "results": results,
    }
