# app/analysis/statistics.py
"""
get_summary_statistics
----------------------
Returns descriptive statistics for every column:
  - Numeric columns  → mean, median, std, min, max, 25th/75th percentiles
  - Categorical/text → top value counts (frequency table)

This is separate from `get_dataset_overview` because overview answers
*"what does the data look like structurally?"* while this answers
*"what are the typical values?"*.
"""

import pandas as pd


def get_summary_statistics(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Compute descriptive statistics for all columns in *df*.

    Parameters
    ----------
    df    : pd.DataFrame – the dataset to describe.
    top_n : int – how many most-frequent values to show for categorical columns.
                  Must be a positive integer.  Defaults to 5.

    Returns
    -------
    dict with two keys:
        "numeric"     – {column: {mean, median, std, min, max, q25, q75}}
        "categorical" – {column: {value: count, ...}} (top_n entries)

    Raises
    ------
    TypeError   if *df* is not a DataFrame, or *top_n* is not an int.
    ValueError  if *df* has no columns, or *top_n* < 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if not isinstance(top_n, int):
        raise TypeError(f"top_n must be an integer, got {type(top_n).__name__}.")
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}.")
    if len(df.columns) == 0:
        raise ValueError("The DataFrame has no columns — nothing to summarise.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    numeric_stats: dict = {}
    for col in numeric_cols:
        series = df[col].dropna()           # ignore NaN for calculations
        if series.empty:
            numeric_stats[col] = {
                "mean": None, "median": None, "std": None,
                "min": None, "max": None, "q25": None, "q75": None,
                "non_null_count": 0,
            }
        else:
            numeric_stats[col] = {
                "mean":   round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std":    round(float(series.std()), 4),
                "min":    round(float(series.min()), 4),
                "max":    round(float(series.max()), 4),
                "q25":    round(float(series.quantile(0.25)), 4),
                "q75":    round(float(series.quantile(0.75)), 4),
                "non_null_count": int(series.count()),
            }

    categorical_stats: dict = {}
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False).head(top_n)
        # Convert index to str so the result is always JSON-serialisable
        categorical_stats[col] = {str(k): int(v) for k, v in counts.items()}

    return {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
    }
