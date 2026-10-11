# app/analysis/overview.py
"""
get_dataset_overview
--------------------
Returns a plain Python dictionary that summarises the *shape* of a DataFrame:
  - how many rows and columns it has
  - each column's data type
  - how many values are missing in each column
  - the number of fully duplicated rows

Why a dict?  Dicts are easy to read, print, and pass to an LLM later.
"""

import pandas as pd


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Return a high-level structural summary of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Any tabular dataset loaded into a DataFrame.

    Returns
    -------
    dict with keys:
        rows          – total number of rows
        columns       – total number of columns
        column_names  – list of column names
        dtypes        – {column: dtype-string} for every column
        missing_counts– {column: int} number of NaN values per column
        missing_pct   – {column: float} percentage of NaN values per column
        duplicate_rows– number of rows that are exact duplicates of another row

    Raises
    ------
    TypeError  if *df* is not a pandas DataFrame.
    ValueError if *df* is completely empty (no columns at all).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty and len(df.columns) == 0:
        raise ValueError("The DataFrame has no columns — nothing to summarise.")

    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {
        col: round(count / n_rows * 100, 2) if n_rows > 0 else 0.0
        for col, count in missing_counts.items()
    }
    duplicate_rows = int(df.duplicated().sum())

    return {
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
    }
