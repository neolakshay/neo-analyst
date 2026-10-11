# app/analysis/visualization.py
"""
generate_visualization
----------------------
Creates an interactive Plotly chart from a DataFrame.

Supported chart types
---------------------
  "bar"       – bar chart (x vs y column)
  "line"      – line chart (x vs y column)
  "scatter"   – scatter plot (x vs y, optional colour-by column)
  "histogram" – distribution of a single numeric column
  "box"       – box-and-whisker plot of one or more columns

The function always returns a `plotly.graph_objects.Figure` object.
You can call `.show()` on it to open it in a browser, or `.write_html()`
to save it as a file.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

VALID_CHART_TYPES = {"bar", "line", "scatter", "histogram", "box"}


def generate_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None = None,
    y_column: str | None = None,
    color_column: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build and return a Plotly Figure from *df*.

    Parameters
    ----------
    df           : pd.DataFrame – the dataset to visualise.
    chart_type   : str – one of "bar", "line", "scatter", "histogram", "box".
    x_column     : column name for the x-axis.
                   Required for bar, line, scatter.
    y_column     : column name for the y-axis.
                   Required for bar, line, scatter.
                   For histogram/box: the column whose distribution to show.
    color_column : (optional) column to use for colour encoding in scatter plots.
    title        : (optional) chart title; a sensible default is generated.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    TypeError   if *df* is not a DataFrame.
    ValueError  if *chart_type* is unknown, required columns are missing,
                or named columns don't exist in *df*.
    """
    # ── Basic input validation ─────────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if not isinstance(chart_type, str):
        raise TypeError(
            f"chart_type must be a string, got {type(chart_type).__name__}."
        )

    chart_type = chart_type.strip().lower()
    if chart_type not in VALID_CHART_TYPES:
        raise ValueError(
            f"Unknown chart type '{chart_type}'. "
            f"Choose from: {sorted(VALID_CHART_TYPES)}"
        )

    # Helper: raise if a column name is given but doesn't exist.
    def _check_col(col_name: str, label: str) -> None:
        if col_name not in df.columns:
            raise ValueError(
                f"{label} column '{col_name}' does not exist. "
                f"Available columns: {list(df.columns)}"
            )

    if color_column is not None:
        _check_col(color_column, "color_column")

    # ── Route to the appropriate chart ────────────────────────────────────────
    if chart_type == "bar":
        if x_column is None or y_column is None:
            raise ValueError("'bar' charts require both x_column and y_column.")
        _check_col(x_column, "x_column")
        _check_col(y_column, "y_column")
        fig = px.bar(
            df,
            x=x_column,
            y=y_column,
            title=title or f"Bar chart: {y_column} by {x_column}",
            color=color_column if color_column else None,
        )

    elif chart_type == "line":
        if x_column is None or y_column is None:
            raise ValueError("'line' charts require both x_column and y_column.")
        _check_col(x_column, "x_column")
        _check_col(y_column, "y_column")
        fig = px.line(
            df,
            x=x_column,
            y=y_column,
            title=title or f"Line chart: {y_column} over {x_column}",
            color=color_column if color_column else None,
        )

    elif chart_type == "scatter":
        if x_column is None or y_column is None:
            raise ValueError("'scatter' charts require both x_column and y_column.")
        _check_col(x_column, "x_column")
        _check_col(y_column, "y_column")
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title or f"Scatter: {y_column} vs {x_column}",
        )

    elif chart_type == "histogram":
        if y_column is None and x_column is None:
            raise ValueError(
                "'histogram' requires x_column (the column whose distribution to show)."
            )
        col = x_column or y_column   # accept either parameter for convenience
        _check_col(col, "x_column")
        fig = px.histogram(
            df,
            x=col,
            title=title or f"Histogram of {col}",
            color=color_column if color_column else None,
        )

    elif chart_type == "box":
        if y_column is None and x_column is None:
            raise ValueError(
                "'box' requires y_column (the numeric column to plot). "
                "Optionally pass x_column to group boxes by a category."
            )
        # y_column is the numeric axis; x_column (optional) is the grouping.
        numeric_col = y_column or x_column
        _check_col(numeric_col, "y_column")
        if x_column and x_column != numeric_col:
            _check_col(x_column, "x_column")
        fig = px.box(
            df,
            x=x_column if x_column != numeric_col else None,
            y=numeric_col,
            title=title or f"Box plot of {numeric_col}",
            color=color_column if color_column else None,
        )

    # Apply a clean, consistent visual theme.
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        title_font_size=16,
    )

    return fig
