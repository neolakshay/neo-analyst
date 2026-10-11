# tests/test_analysis.py
"""
Unit tests for the five analytical functions.

Each test class groups tests for one function.
Each test method is named test_<what_is_being_tested>.

Run with:  pytest tests/ -v
"""

import math
import pandas as pd
import pytest

from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. get_dataset_overview
# ══════════════════════════════════════════════════════════════════════════════

class TestGetDatasetOverview:

    def test_returns_dict(self, sample_df):
        result = get_dataset_overview(sample_df)
        assert isinstance(result, dict)

    def test_correct_row_and_column_count(self, sample_df):
        result = get_dataset_overview(sample_df)
        assert result["rows"] == 20
        assert result["columns"] == 8

    def test_column_names_match(self, sample_df):
        result = get_dataset_overview(sample_df)
        assert result["column_names"] == list(sample_df.columns)

    def test_missing_counts_are_correct(self, sample_df):
        result = get_dataset_overview(sample_df)
        # Row 19 has no grade; row 20 has no score and no study_hours.
        assert result["missing_counts"]["grade"] == 1
        assert result["missing_counts"]["score"] == 1
        assert result["missing_counts"]["study_hours"] == 1

    def test_missing_percentage_calculation(self, sample_df):
        result = get_dataset_overview(sample_df)
        # 1 missing out of 20 rows = 5.0 %
        assert result["missing_pct"]["grade"] == 5.0

    def test_no_duplicates_in_sample(self, sample_df):
        result = get_dataset_overview(sample_df)
        assert result["duplicate_rows"] == 0

    def test_detects_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = get_dataset_overview(df)
        assert result["duplicate_rows"] == 1

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            get_dataset_overview([1, 2, 3])

    def test_raises_on_empty_dataframe_no_columns(self):
        with pytest.raises(ValueError, match="no columns"):
            get_dataset_overview(pd.DataFrame())

    def test_works_on_dataframe_with_rows_but_missing_values(self, df_with_missing):
        result = get_dataset_overview(df_with_missing)
        assert result["rows"] == 5
        assert result["missing_counts"]["a"] == 2
        assert result["missing_counts"]["b"] == 1

    def test_dtypes_are_strings(self, clean_df):
        result = get_dataset_overview(clean_df)
        for dtype_str in result["dtypes"].values():
            assert isinstance(dtype_str, str)


# ══════════════════════════════════════════════════════════════════════════════
# 2. get_summary_statistics
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSummaryStatistics:

    def test_returns_dict_with_correct_keys(self, clean_df):
        result = get_summary_statistics(clean_df)
        assert "numeric" in result
        assert "categorical" in result

    def test_numeric_stats_keys_present(self, clean_df):
        result = get_summary_statistics(clean_df)
        expected_keys = {"mean", "median", "std", "min", "max", "q25", "q75", "non_null_count"}
        for col, stats in result["numeric"].items():
            assert set(stats.keys()) == expected_keys, f"Missing keys for column '{col}'"

    def test_mean_is_correct(self, clean_df):
        result = get_summary_statistics(clean_df)
        # x = [1,2,3,4,5] → mean = 3.0
        assert result["numeric"]["x"]["mean"] == pytest.approx(3.0, abs=1e-4)

    def test_median_is_correct(self, clean_df):
        result = get_summary_statistics(clean_df)
        assert result["numeric"]["x"]["median"] == pytest.approx(3.0, abs=1e-4)

    def test_std_is_correct(self, clean_df):
        result = get_summary_statistics(clean_df)
        # Sample std of [1,2,3,4,5] ≈ 1.5811
        assert result["numeric"]["x"]["std"] == pytest.approx(1.5811, abs=1e-3)

    def test_min_max_are_correct(self, clean_df):
        result = get_summary_statistics(clean_df)
        assert result["numeric"]["x"]["min"] == 1.0
        assert result["numeric"]["x"]["max"] == 5.0

    def test_categorical_frequency(self, clean_df):
        result = get_summary_statistics(clean_df)
        # category = [A, B, A, B, A] → A:3, B:2
        cat = result["categorical"]["category"]
        assert cat["A"] == 3
        assert cat["B"] == 2

    def test_top_n_limits_categorical_output(self, sample_df):
        result = get_summary_statistics(sample_df, top_n=2)
        for col, counts in result["categorical"].items():
            assert len(counts) <= 2

    def test_handles_all_null_numeric_column(self):
        # Use float NaN so pandas infers a float64 (numeric) column.
        # Python None alone causes pandas to pick object dtype.
        import numpy as np
        df = pd.DataFrame({"vals": [np.nan, np.nan, np.nan]})
        result = get_summary_statistics(df)
        stats = result["numeric"]["vals"]
        assert stats["mean"] is None
        assert stats["non_null_count"] == 0

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            get_summary_statistics("not a dataframe")

    def test_raises_on_zero_top_n(self, clean_df):
        with pytest.raises(ValueError, match="top_n"):
            get_summary_statistics(clean_df, top_n=0)

    def test_raises_on_bad_top_n_type(self, clean_df):
        with pytest.raises(TypeError, match="top_n"):
            get_summary_statistics(clean_df, top_n="five")

    def test_non_null_count_excludes_nans(self, df_with_missing):
        result = get_summary_statistics(df_with_missing)
        # column 'a' has 2 NaN values out of 5
        assert result["numeric"]["a"]["non_null_count"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 3. filter_data
# ══════════════════════════════════════════════════════════════════════════════

class TestFilterData:

    def test_eq_operator(self, sample_df):
        result = filter_data(sample_df, [{"column": "grade", "operator": "eq", "value": "A"}])
        assert all(result["grade"] == "A")
        # Alice(1), Carol(3), Grace(7), Karl(11), Nora(14), Rosa(18) → 6 rows (Sam row 19 has NaN grade)
        assert len(result) == 6

    def test_gt_operator(self, clean_df):
        result = filter_data(clean_df, [{"column": "x", "operator": "gt", "value": 3}])
        assert list(result["x"]) == [4.0, 5.0]

    def test_gte_operator(self, clean_df):
        result = filter_data(clean_df, [{"column": "x", "operator": "gte", "value": 3}])
        assert list(result["x"]) == [3.0, 4.0, 5.0]

    def test_lt_operator(self, clean_df):
        result = filter_data(clean_df, [{"column": "x", "operator": "lt", "value": 3}])
        assert list(result["x"]) == [1.0, 2.0]

    def test_lte_operator(self, clean_df):
        result = filter_data(clean_df, [{"column": "x", "operator": "lte", "value": 3}])
        assert list(result["x"]) == [1.0, 2.0, 3.0]

    def test_ne_operator(self, clean_df):
        result = filter_data(clean_df, [{"column": "category", "operator": "ne", "value": "A"}])
        assert all(result["category"] == "B")

    def test_contains_operator(self, sample_df):
        result = filter_data(sample_df, [{"column": "city", "operator": "contains", "value": "lon"}])
        assert all(result["city"] == "London")

    def test_contains_is_case_insensitive(self, sample_df):
        lower = filter_data(sample_df, [{"column": "city", "operator": "contains", "value": "london"}])
        upper = filter_data(sample_df, [{"column": "city", "operator": "contains", "value": "LONDON"}])
        assert len(lower) == len(upper)

    def test_isnull_operator(self, sample_df):
        result = filter_data(sample_df, [{"column": "grade", "operator": "isnull"}])
        assert len(result) == 1   # only row 19 (Sam) has a null grade
        assert result.iloc[0]["name"] == "Sam"

    def test_notnull_operator(self, sample_df):
        result = filter_data(sample_df, [{"column": "score", "operator": "notnull"}])
        assert len(result) == 19  # 1 row has null score

    def test_multiple_conditions_combined_with_and(self, sample_df):
        result = filter_data(sample_df, [
            {"column": "score",  "operator": "gte", "value": 80},
            {"column": "passed", "operator": "eq",  "value": True},
        ])
        assert all(result["score"] >= 80)
        assert all(result["passed"] == True)

    def test_empty_result_when_no_rows_match(self, clean_df):
        result = filter_data(clean_df, [{"column": "x", "operator": "gt", "value": 999}])
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_original_df_is_not_modified(self, clean_df):
        original_len = len(clean_df)
        filter_data(clean_df, [{"column": "x", "operator": "gt", "value": 3}])
        assert len(clean_df) == original_len

    def test_raises_on_missing_column_key(self, clean_df):
        with pytest.raises(ValueError, match="'column' key"):
            filter_data(clean_df, [{"operator": "eq", "value": 1}])

    def test_raises_on_unknown_column(self, clean_df):
        with pytest.raises(ValueError, match="does not exist"):
            filter_data(clean_df, [{"column": "nonexistent", "operator": "eq", "value": 1}])

    def test_raises_on_unknown_operator(self, clean_df):
        with pytest.raises(ValueError, match="Unknown operator"):
            filter_data(clean_df, [{"column": "x", "operator": "between", "value": 1}])

    def test_raises_on_missing_value_for_operator_that_needs_it(self, clean_df):
        with pytest.raises(ValueError, match="requires a 'value' key"):
            filter_data(clean_df, [{"column": "x", "operator": "gt"}])

    def test_raises_on_empty_conditions(self, clean_df):
        with pytest.raises(ValueError, match="empty"):
            filter_data(clean_df, [])

    def test_raises_on_non_list_conditions(self, clean_df):
        with pytest.raises(TypeError, match="list"):
            filter_data(clean_df, {"column": "x", "operator": "eq", "value": 1})


# ══════════════════════════════════════════════════════════════════════════════
# 4. analyze_correlations
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeCorrelations:

    def test_returns_dict_with_expected_keys(self, clean_df):
        result = analyze_correlations(clean_df)
        assert set(result.keys()) == {
            "columns_used", "matrix", "top_positive", "top_negative", "warning"
        }

    def test_perfect_positive_correlation(self, clean_df):
        # y = 2*x, so x↔y should be r = 1.0
        result = analyze_correlations(clean_df, columns=["x", "y"])
        assert result["matrix"]["x"]["y"] == pytest.approx(1.0, abs=1e-4)

    def test_perfect_negative_correlation(self, clean_df):
        # z = 12 - 2*x, so x↔z should be r = -1.0
        result = analyze_correlations(clean_df, columns=["x", "z"])
        assert result["matrix"]["x"]["z"] == pytest.approx(-1.0, abs=1e-4)

    def test_diagonal_is_one(self, clean_df):
        result = analyze_correlations(clean_df)
        for col in result["columns_used"]:
            assert result["matrix"][col][col] == pytest.approx(1.0, abs=1e-4)

    def test_top_positive_is_sorted_descending(self, clean_df):
        result = analyze_correlations(clean_df)
        rs = [r for _, _, r in result["top_positive"]]
        assert rs == sorted(rs, reverse=True)

    def test_top_negative_is_sorted_ascending(self, clean_df):
        result = analyze_correlations(clean_df)
        rs = [r for _, _, r in result["top_negative"]]
        assert rs == sorted(rs)

    def test_columns_used_excludes_categoricals_by_default(self, clean_df):
        result = analyze_correlations(clean_df)
        # 'category' column is not numeric, should be excluded
        assert "category" not in result["columns_used"]

    def test_explicit_columns_parameter(self, clean_df):
        result = analyze_correlations(clean_df, columns=["x", "y"])
        assert result["columns_used"] == ["x", "y"]

    def test_warning_when_fewer_than_two_numeric_columns(self, single_col_df):
        result = analyze_correlations(single_col_df)
        assert result["warning"] is not None
        assert result["matrix"] == {}

    def test_raises_on_non_numeric_column_in_explicit_list(self, clean_df):
        with pytest.raises(ValueError, match="not numeric"):
            analyze_correlations(clean_df, columns=["x", "category"])

    def test_raises_on_nonexistent_column(self, clean_df):
        with pytest.raises(ValueError, match="does not exist"):
            analyze_correlations(clean_df, columns=["x", "missing_col"])

    def test_raises_on_bad_top_n(self, clean_df):
        with pytest.raises(ValueError, match="top_n"):
            analyze_correlations(clean_df, top_n=0)

    def test_works_with_sample_df(self, sample_df):
        # Should run without error; study_hours and score should be positively correlated.
        result = analyze_correlations(sample_df)
        assert result["warning"] is None
        top = result["top_positive"][0]
        # top[0], top[1] are column names; top[2] is r
        assert top[2] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. generate_visualization
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateVisualization:
    """
    We test that the function returns a Plotly Figure with the right title
    and that invalid inputs raise the right errors.
    We do NOT test visual appearance (that would require a browser).
    """

    def test_scatter_returns_figure(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "scatter", x_column="x", y_column="y")
        assert isinstance(fig, go.Figure)

    def test_bar_returns_figure(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "bar", x_column="category", y_column="x")
        assert isinstance(fig, go.Figure)

    def test_line_returns_figure(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "line", x_column="x", y_column="y")
        assert isinstance(fig, go.Figure)

    def test_histogram_returns_figure(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "histogram", x_column="x")
        assert isinstance(fig, go.Figure)

    def test_box_returns_figure(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "box", y_column="x")
        assert isinstance(fig, go.Figure)

    def test_custom_title_is_used(self, clean_df):
        fig = generate_visualization(
            clean_df, "scatter", x_column="x", y_column="y", title="My Custom Title"
        )
        assert fig.layout.title.text == "My Custom Title"

    def test_default_title_is_generated(self, clean_df):
        fig = generate_visualization(clean_df, "scatter", x_column="x", y_column="y")
        assert fig.layout.title.text is not None
        assert len(fig.layout.title.text) > 0

    def test_chart_type_is_case_insensitive(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "SCATTER", x_column="x", y_column="y")
        assert isinstance(fig, go.Figure)

    def test_raises_on_unknown_chart_type(self, clean_df):
        with pytest.raises(ValueError, match="Unknown chart type"):
            generate_visualization(clean_df, "pie", x_column="x", y_column="y")

    def test_raises_when_x_column_missing_for_scatter(self, clean_df):
        with pytest.raises(ValueError, match="require"):
            generate_visualization(clean_df, "scatter", y_column="y")

    def test_raises_when_y_column_missing_for_bar(self, clean_df):
        with pytest.raises(ValueError, match="require"):
            generate_visualization(clean_df, "bar", x_column="x")

    def test_raises_on_nonexistent_x_column(self, clean_df):
        with pytest.raises(ValueError, match="does not exist"):
            generate_visualization(clean_df, "scatter", x_column="ghost", y_column="y")

    def test_raises_on_nonexistent_y_column(self, clean_df):
        with pytest.raises(ValueError, match="does not exist"):
            generate_visualization(clean_df, "scatter", x_column="x", y_column="ghost")

    def test_raises_on_non_dataframe_input(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            generate_visualization({"a": [1, 2]}, "scatter", x_column="a", y_column="a")

    def test_histogram_accepts_x_column(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(clean_df, "histogram", x_column="x")
        assert isinstance(fig, go.Figure)

    def test_box_with_grouping(self, clean_df):
        import plotly.graph_objects as go
        fig = generate_visualization(
            clean_df, "box", x_column="category", y_column="x"
        )
        assert isinstance(fig, go.Figure)
