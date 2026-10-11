# app/analysis/filtering.py
"""
filter_data
-----------
Filters rows of a DataFrame based on a list of conditions.

Each condition is a small dict like:
    {"column": "age", "operator": "gt", "value": 30}

Supported operators
-------------------
  eq       – equal to
  ne       – not equal to
  gt       – greater than
  gte      – greater than or equal to
  lt       – less than
  lte      – less than or equal to
  contains – string/substring match (case-insensitive)
  isnull   – value is NaN/None  (no 'value' key needed)
  notnull  – value is not NaN   (no 'value' key needed)

Multiple conditions are combined with AND (all must be true).
"""

from typing import Any
import pandas as pd

# All operators we recognise.
VALID_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "isnull", "notnull"}

# Operators that don't need a comparison value.
NULL_OPERATORS = {"isnull", "notnull"}


def _validate_condition(condition: dict, df: pd.DataFrame) -> None:
    """Raise a clear error if a condition dict is malformed."""
    if not isinstance(condition, dict):
        raise TypeError(f"Each condition must be a dict, got {type(condition).__name__}.")

    if "column" not in condition:
        raise ValueError(f"Condition is missing the 'column' key: {condition}")

    column = condition["column"]
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    if "operator" not in condition:
        raise ValueError(f"Condition is missing the 'operator' key: {condition}")

    operator = condition["operator"]
    if operator not in VALID_OPERATORS:
        raise ValueError(
            f"Unknown operator '{operator}'. "
            f"Valid operators are: {sorted(VALID_OPERATORS)}"
        )

    if operator not in NULL_OPERATORS and "value" not in condition:
        raise ValueError(
            f"Operator '{operator}' requires a 'value' key in the condition."
        )


def _apply_condition(df: pd.DataFrame, condition: dict) -> "pd.Series[bool]":
    """Return a boolean Series: True for rows that satisfy *condition*."""
    col: str = condition["column"]
    op: str = condition["operator"]
    val: Any = condition.get("value")
    series = df[col]

    if op == "eq":
        return series == val
    if op == "ne":
        return series != val
    if op == "gt":
        return series > val
    if op == "gte":
        return series >= val
    if op == "lt":
        return series < val
    if op == "lte":
        return series <= val
    if op == "contains":
        # Works on string columns; cast to str first to be safe. Use regex=False for literal substring matching.
        return series.astype(str).str.contains(str(val), case=False, na=False, regex=False)
    if op == "isnull":
        return series.isnull()
    if op == "notnull":
        return series.notnull()

    # Should never reach here because _validate_condition checks the operator.
    raise ValueError(f"Unhandled operator: {op}")  # pragma: no cover


def filter_data(df: pd.DataFrame, conditions: list[dict]) -> pd.DataFrame:
    """Filter rows of *df* using a list of conditions (all must match).

    Parameters
    ----------
    df         : pd.DataFrame – the dataset to filter.
    conditions : list[dict]   – one or more condition dicts (see module docs).

    Returns
    -------
    pd.DataFrame – a new DataFrame containing only the matching rows.
                   The original index is preserved.  May be empty if nothing matches.

    Raises
    ------
    TypeError   if *df* is not a DataFrame or *conditions* is not a list.
    ValueError  if any condition is structurally invalid or references a missing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if not isinstance(conditions, list):
        raise TypeError(
            f"'conditions' must be a list of dicts, got {type(conditions).__name__}."
        )
    if len(conditions) == 0:
        raise ValueError(
            "'conditions' is empty. Provide at least one filter condition."
        )

    # Validate every condition before applying any of them.
    for condition in conditions:
        _validate_condition(condition, df)

    # Build a combined boolean mask (AND logic).
    mask = pd.Series([True] * len(df), index=df.index)
    for condition in conditions:
        mask = mask & _apply_condition(df, condition)

    return df[mask].copy()
