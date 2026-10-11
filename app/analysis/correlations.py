# app/analysis/correlations.py
"""
analyze_correlations
--------------------
Computes the Pearson correlation matrix for all numeric columns and surfaces
the strongest (positive and negative) pairs.

Pearson correlation ranges from -1 to +1:
  +1  → perfect positive relationship (both go up together)
  -1  → perfect negative relationship (one goes up, the other goes down)
   0  → no linear relationship

Only numeric columns participate; categorical columns are silently ignored.
"""

import pandas as pd


def analyze_correlations(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    top_n: int = 5,
    min_periods: int = 1,
) -> dict:
    """Compute pairwise Pearson correlations for numeric columns.

    Parameters
    ----------
    df          : pd.DataFrame – the dataset.
    columns     : optional list of column names to include.
                  If None, all numeric columns are used.
    top_n       : how many strongest pairs to highlight (default 5).
    min_periods : minimum number of valid (non-NaN) observations required
                  per pair to compute a correlation (default 1).

    Returns
    -------
    dict with keys:
        "columns_used"   – list of columns that were included
        "matrix"         – {col: {col: correlation}} – the full correlation matrix
        "top_positive"   – [(col_a, col_b, r)] top positive pairs, sorted desc
        "top_negative"   – [(col_a, col_b, r)] top negative pairs, sorted asc
        "warning"        – string or None – e.g. if fewer than 2 numeric cols

    Raises
    ------
    TypeError   if *df* is not a DataFrame.
    ValueError  if any name in *columns* does not exist or is not numeric.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}.")

    # ── Determine which columns to use ────────────────────────────────────────
    if columns is not None:
        # Validate every requested column.
        for col in columns:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' does not exist. "
                    f"Available columns: {list(df.columns)}"
                )
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Column '{col}' is not numeric "
                    f"(dtype: {df[col].dtype}). "
                    "Only numeric columns can be correlated."
                )
        numeric_cols = list(columns)
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ── Guard: need at least 2 columns ────────────────────────────────────────
    if len(numeric_cols) < 2:
        return {
            "columns_used": numeric_cols,
            "matrix": {},
            "top_positive": [],
            "top_negative": [],
            "warning": (
                "Need at least 2 numeric columns to compute correlations. "
                f"Found: {numeric_cols}"
            ),
        }

    # ── Compute the correlation matrix ────────────────────────────────────────
    corr_matrix = df[numeric_cols].corr(method="pearson", min_periods=min_periods)

    # Convert to nested plain-Python dict (round to 4 dp for readability).
    matrix_dict: dict = {
        col: {
            other: round(float(val), 4)
            for other, val in row.items()
        }
        for col, row in corr_matrix.to_dict().items()
    }

    # ── Extract unique pairs (upper triangle, excluding diagonal) ─────────────
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            r = corr_matrix.loc[col_a, col_b]
            if pd.notna(r):
                pairs.append((col_a, col_b, round(float(r), 4)))

    # Sort descending for positive pairs, ascending for negative.
    top_positive = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]
    top_negative = sorted(pairs, key=lambda x: x[2])[:top_n]

    return {
        "columns_used": numeric_cols,
        "matrix": matrix_dict,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "warning": None,
    }
