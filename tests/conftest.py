# tests/conftest.py
"""
Shared pytest fixtures.

A fixture is a function decorated with @pytest.fixture.
pytest automatically injects fixtures into test functions that request them
by name.  This keeps every test function short and focused.
"""

import pathlib
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_path() -> pathlib.Path:
    """Absolute path to examples/sample_data.csv."""
    return pathlib.Path(__file__).resolve().parent.parent / "examples" / "sample_data.csv"


@pytest.fixture
def sample_df(sample_csv_path) -> pd.DataFrame:
    """The sample CSV loaded as a DataFrame (raw, no preprocessing)."""
    return pd.read_csv(sample_csv_path)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, fully controlled DataFrame with no missing values.

    Useful for tests that need predictable numerical results.
    """
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],   # perfect positive correlation with x
        "z": [10.0, 8.0, 6.0, 4.0, 2.0],   # perfect negative correlation with x
        "category": ["A", "B", "A", "B", "A"],
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """DataFrame that deliberately contains NaN values."""
    return pd.DataFrame({
        "a": [1.0, None, 3.0, None, 5.0],
        "b": ["x", "y", None, "y", "x"],
    })


@pytest.fixture
def single_col_df() -> pd.DataFrame:
    """DataFrame with only one numeric column (edge case for correlations)."""
    return pd.DataFrame({"value": [10, 20, 30]})
