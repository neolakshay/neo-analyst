# tests/test_analysis.py
"""
Unit tests for the five analytical functions.

Each test class groups tests for one function.
Each test method is named test_<what_is_being_tested>.

Run with:  pytest tests/ -v
"""

import json
import math
from pathlib import Path
import pandas as pd
import pytest

from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
    group_and_aggregate,
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
            "columns_used", "matrix", "pair_counts", "top_positive", "top_negative", "warning"
        }

    def test_pair_counts_generalization_dataset(self):
        """Verify exact pairwise complete observation counts on generalization dataset."""
        import pathlib
        gen_path = pathlib.Path(__file__).resolve().parent.parent / "examples" / "generalization_test.csv"
        df = pd.read_csv(gen_path)
        result = analyze_correlations(df)

        counts = result["pair_counts"]
        assert counts["years_experience"]["monthly_salary"] == 14
        assert counts["years_experience"]["satisfaction_score"] == 14
        assert counts["monthly_salary"]["satisfaction_score"] == 13

        # Symmetry
        assert counts["monthly_salary"]["years_experience"] == 14
        assert counts["satisfaction_score"]["years_experience"] == 14
        assert counts["satisfaction_score"]["monthly_salary"] == 13

        # Pearson coefficients computed correctly with pairwise deletion
        exp_corr = df[["years_experience", "monthly_salary", "satisfaction_score"]].corr(method="pearson")
        assert result["matrix"]["years_experience"]["monthly_salary"] == pytest.approx(
            exp_corr.loc["years_experience", "monthly_salary"], abs=1e-4
        )
        assert result["matrix"]["monthly_salary"]["satisfaction_score"] == pytest.approx(
            exp_corr.loc["monthly_salary", "satisfaction_score"], abs=1e-4
        )

    def test_pair_counts_symmetry_and_types(self, clean_df):
        """Pair counts must be non-negative integers and symmetric across variables."""
        result = analyze_correlations(clean_df)
        counts = result["pair_counts"]
        for col_a in result["columns_used"]:
            for col_b in result["columns_used"]:
                val = counts[col_a][col_b]
                assert isinstance(val, int)
                assert val == counts[col_b][col_a]
                assert val == len(clean_df)  # clean_df has no missing values

    def test_pair_counts_single_numeric_column_warning(self, single_col_df):
        """When fewer than 2 numeric columns exist, pair_counts is an empty dict."""
        result = analyze_correlations(single_col_df)
        assert result["pair_counts"] == {}
        assert result["matrix"] == {}
        assert result["warning"] is not None

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

    def test_raises_on_non_string_chart_type(self, clean_df):
        with pytest.raises(TypeError, match="chart_type must be a string"):
            generate_visualization(clean_df, 123, x_column="x", y_column="y")

    def test_raises_on_nonexistent_color_column_bar(self, clean_df):
        with pytest.raises(ValueError, match="color_column column 'ghost' does not exist"):
            generate_visualization(
                clean_df, "bar", x_column="category", y_column="x", color_column="ghost"
            )

    def test_raises_on_nonexistent_color_column_line(self, clean_df):
        with pytest.raises(ValueError, match="color_column column 'ghost' does not exist"):
            generate_visualization(
                clean_df, "line", x_column="x", y_column="y", color_column="ghost"
            )

    def test_raises_on_nonexistent_color_column_histogram(self, clean_df):
        with pytest.raises(ValueError, match="color_column column 'ghost' does not exist"):
            generate_visualization(
                clean_df, "histogram", x_column="x", color_column="ghost"
            )

    def test_raises_on_nonexistent_color_column_box(self, clean_df):
        with pytest.raises(ValueError, match="color_column column 'ghost' does not exist"):
            generate_visualization(
                clean_df, "box", y_column="x", color_column="ghost"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Regression tests for bug fixes & edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestRegressionAndEdgeCases:

    def test_single_non_null_numeric_std_is_none(self):
        """std of a single observation is undefined (ddof=1); should be None, not float('nan')."""
        df = pd.DataFrame({"single_val": [42.0, None, None]})
        stats = get_summary_statistics(df)
        col_stats = stats["numeric"]["single_val"]
        assert col_stats["mean"] == 42.0
        assert col_stats["std"] is None
        assert col_stats["non_null_count"] == 1

    def test_contains_with_regex_special_characters(self):
        """contains operator must perform literal substring match and not fail on regex chars."""
        df = pd.DataFrame({
            "text": ["4K (UHD) Monitor", "Price + Tax", "Question?", "Item [A*]"]
        })
        res1 = filter_data(df, [{"column": "text", "operator": "contains", "value": "4K ("}])
        assert len(res1) == 1
        assert res1.iloc[0]["text"] == "4K (UHD) Monitor"

        res2 = filter_data(df, [{"column": "text", "operator": "contains", "value": "[A*]"}])
        assert len(res2) == 1
        assert res2.iloc[0]["text"] == "Item [A*]"

    def test_constant_column_correlation_returns_none_and_warning(self):
        """Constant columns have zero variance; Pearson correlation is undefined (None, not NaN)."""
        df = pd.DataFrame({"const_a": [5, 5, 5], "const_b": [10, 10, 10]})
        result = analyze_correlations(df)
        assert result["matrix"]["const_a"]["const_b"] is None
        assert result["matrix"]["const_a"]["const_a"] is None
        assert result["top_positive"] == []
        assert result["top_negative"] == []
        assert result["warning"] is not None

    def test_top_positive_only_includes_positive_correlations(self, clean_df):
        """Ensure top_positive contains only pairs with r > 0."""
        result = analyze_correlations(clean_df)
        for _, _, r in result["top_positive"]:
            assert r > 0

    def test_top_negative_only_includes_negative_correlations(self, clean_df):
        """Ensure top_negative contains only pairs with r < 0."""
        result = analyze_correlations(clean_df)
        for _, _, r in result["top_negative"]:
            assert r < 0

    def test_datasets_with_no_negative_correlations_returns_empty_top_negative(self, employee_df):
        """employee_analytics has only positive pairwise correlations; top_negative must be empty."""
        result = analyze_correlations(employee_df)
        assert len(result["top_negative"]) == 0
        assert len(result["top_positive"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 7. Multi-dataset validation suite
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiDatasetValidation:

    def test_ecommerce_dataset_full_pipeline(self, ecommerce_df):
        # 1. Overview
        ov = get_dataset_overview(ecommerce_df)
        assert ov["rows"] == 150
        assert ov["columns"] == 7
        assert ov["missing_counts"]["quantity"] == 7
        assert ov["missing_counts"]["unit_price"] == 5
        assert ov["missing_counts"]["discount_pct"] == 33

        # 2. Statistics
        stats = get_summary_statistics(ecommerce_df, top_n=3)
        assert "quantity" in stats["numeric"]
        assert stats["numeric"]["quantity"]["non_null_count"] == 143
        assert stats["numeric"]["unit_price"]["min"] == 45.0
        assert stats["numeric"]["unit_price"]["max"] == 85000.0

        # 3. Filter
        filtered = filter_data(ecommerce_df, [
            {"column": "category", "operator": "eq", "value": "Electronics"},
            {"column": "unit_price", "operator": "gte", "value": 5000},
            {"column": "discount_pct", "operator": "notnull"},
        ])
        assert len(filtered) > 0
        assert all(filtered["category"] == "Electronics")
        assert all(filtered["unit_price"] >= 5000)
        assert all(filtered["discount_pct"].notnull())

        # 4. Correlations
        corr = analyze_correlations(ecommerce_df)
        assert "unit_price" in corr["matrix"]
        assert len(corr["top_positive"]) > 0
        assert len(corr["top_negative"]) > 0

        # 5. Visualizations
        fig_scatter = generate_visualization(
            ecommerce_df, "scatter", x_column="quantity", y_column="unit_price", color_column="category"
        )
        assert fig_scatter is not None
        fig_hist = generate_visualization(ecommerce_df, "histogram", x_column="unit_price")
        assert fig_hist is not None

    def test_employee_dataset_full_pipeline(self, employee_df):
        # 1. Overview
        ov = get_dataset_overview(employee_df)
        assert ov["rows"] == 50
        assert ov["columns"] == 6
        assert ov["missing_counts"]["department"] == 2
        assert ov["missing_counts"]["salary"] == 4

        # 2. Statistics
        stats = get_summary_statistics(employee_df)
        assert stats["numeric"]["salary"]["mean"] == pytest.approx(1486086.9565, abs=1e-2)
        assert stats["numeric"]["experience_years"]["median"] == 7.0

        # 3. Filter
        filtered = filter_data(employee_df, [
            {"column": "department", "operator": "eq", "value": "Engineering"},
            {"column": "remote", "operator": "eq", "value": "Yes"},
        ])
        assert len(filtered) == 5
        assert all(filtered["department"] == "Engineering")
        assert all(filtered["remote"] == "Yes")

        # 4. Correlations
        corr = analyze_correlations(employee_df)
        assert corr["matrix"]["experience_years"]["salary"] == pytest.approx(0.8525, abs=1e-3)
        assert len(corr["top_positive"]) == 3
        assert len(corr["top_negative"]) == 0

        # 5. Visualizations
        fig_box = generate_visualization(
            employee_df, "box", x_column="department", y_column="salary"
        )
        assert fig_box is not None

    def test_edge_cases_dataset_full_pipeline(self, edge_cases_df):
        # 1. Overview
        ov = get_dataset_overview(edge_cases_df)
        assert ov["rows"] == 20
        assert ov["columns"] == 5
        assert ov["missing_counts"]["score"] == 4
        assert ov["missing_counts"]["bonus"] == 4

        # 2. Statistics
        stats = get_summary_statistics(edge_cases_df)
        assert stats["numeric"]["score"]["max"] == 999999.0
        assert stats["numeric"]["score"]["min"] == -15.0
        assert stats["numeric"]["score"]["non_null_count"] == 16

        # 3. Filter
        f_null = filter_data(edge_cases_df, [{"column": "score", "operator": "isnull"}])
        assert len(f_null) == 4
        f_outlier = filter_data(edge_cases_df, [{"column": "notes", "operator": "contains", "value": "outlier"}])
        assert len(f_outlier) == 1

        # 4. Correlations
        corr = analyze_correlations(edge_cases_df)
        assert corr["matrix"]["score"]["bonus"] == pytest.approx(1.0, abs=1e-4)

        # 5. Visualizations
        fig_bar = generate_visualization(
            edge_cases_df, "bar", x_column="id", y_column="score"
        )
        assert fig_bar is not None


# ══════════════════════════════════════════════════════════════════════════════
# 8. group_and_aggregate tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGroupAndAggregate:

    @pytest.fixture
    def employees_sample(self) -> pd.DataFrame:
        """Controlled synthetic employee DataFrame."""
        return pd.DataFrame({
            "department": ["Engineering", "Engineering", "Marketing", "Marketing", "Sales"],
            "salary": [100000.0, 150000.0, 80000.0, 90000.0, 70000.0],
            "bonus": [10000.0, 15000.0, None, 5000.0, 7000.0],
            "level": ["Senior", "Staff", "Junior", "Senior", "Junior"],
        })

    def test_group_by_one_categorical_and_calculate_mean(self, employees_sample):
        result = group_and_aggregate(
            employees_sample,
            group_by=["department"],
            aggregations={"salary": ["mean"]},
        )
        assert result["group_by"] == ["department"]
        assert result["aggregations"] == {"salary": ["mean"]}
        assert result["group_count"] == 3

        # Lookup by department
        eng = next(r for r in result["results"] if r["department"] == "Engineering")
        assert eng["salary_mean"] == 125000.0
        mkt = next(r for r in result["results"] if r["department"] == "Marketing")
        assert mkt["salary_mean"] == 85000.0
        sales = next(r for r in result["results"] if r["department"] == "Sales")
        assert sales["salary_mean"] == 70000.0

    def test_group_by_one_column_multiple_operations(self, employees_sample):
        result = group_and_aggregate(
            employees_sample,
            group_by=["department"],
            aggregations={"salary": ["mean", "median", "sum", "min", "max", "count"]},
        )
        eng = next(r for r in result["results"] if r["department"] == "Engineering")
        assert eng["salary_mean"] == 125000.0
        assert eng["salary_median"] == 125000.0
        assert eng["salary_sum"] == 250000.0
        assert eng["salary_min"] == 100000.0
        assert eng["salary_max"] == 150000.0
        assert eng["salary_count"] == 2
        assert isinstance(eng["salary_count"], int)

    def test_group_by_one_column_aggregate_multiple_numeric_columns(self, employees_sample):
        result = group_and_aggregate(
            employees_sample,
            group_by=["department"],
            aggregations={
                "salary": ["mean"],
                "bonus": ["sum", "count"],
            },
        )
        eng = next(r for r in result["results"] if r["department"] == "Engineering")
        assert "salary_mean" in eng
        assert "bonus_sum" in eng
        assert "bonus_count" in eng
        assert eng["bonus_sum"] == 25000.0
        assert eng["bonus_count"] == 2

    def test_group_by_multiple_columns(self, employees_sample):
        result = group_and_aggregate(
            employees_sample,
            group_by=["department", "level"],
            aggregations={"salary": ["mean"]},
        )
        assert result["group_by"] == ["department", "level"]
        # 5 distinct (department, level) pairs
        assert result["group_count"] == 5
        eng_senior = next(
            r for r in result["results"]
            if r["department"] == "Engineering" and r["level"] == "Senior"
        )
        assert eng_senior["salary_mean"] == 100000.0

    def test_count_non_null_values_with_missing_values(self, employees_sample):
        result = group_and_aggregate(
            employees_sample,
            group_by=["department"],
            aggregations={"bonus": ["count"]},
        )
        mkt = next(r for r in result["results"] if r["department"] == "Marketing")
        # Marketing has 2 rows but only 1 non-null bonus
        assert mkt["bonus_count"] == 1

    def test_missing_grouping_values_dropna_true(self):
        df = pd.DataFrame({
            "dept": ["Eng", "Eng", None],
            "salary": [100.0, 200.0, 300.0],
        })
        result = group_and_aggregate(
            df, group_by=["dept"], aggregations={"salary": ["mean"]}, dropna=True
        )
        assert result["group_count"] == 1
        assert result["results"][0]["dept"] == "Eng"
        assert result["results"][0]["salary_mean"] == 150.0

    def test_missing_grouping_values_dropna_false(self):
        df = pd.DataFrame({
            "dept": ["Eng", "Eng", None],
            "salary": [100.0, 200.0, 300.0],
        })
        result = group_and_aggregate(
            df, group_by=["dept"], aggregations={"salary": ["mean"]}, dropna=False
        )
        assert result["group_count"] == 2
        null_group = next(r for r in result["results"] if r["dept"] is None)
        assert null_group["salary_mean"] == 300.0

    def test_no_matching_non_null_groups_returns_documented_structure(self):
        df = pd.DataFrame({
            "dept": [None, None],
            "salary": [100.0, 200.0],
        })
        result = group_and_aggregate(
            df, group_by=["dept"], aggregations={"salary": ["mean"]}, dropna=True
        )
        assert result["group_by"] == ["dept"]
        assert result["aggregations"] == {"salary": ["mean"]}
        assert result["group_count"] == 0
        assert result["results"] == []

    def test_empty_dataframe_with_valid_columns(self):
        df = pd.DataFrame(columns=["dept", "salary"])
        result = group_and_aggregate(
            df, group_by=["dept"], aggregations={"salary": ["mean"]}
        )
        assert result["group_by"] == ["dept"]
        assert result["group_count"] == 0
        assert result["results"] == []

    def test_reject_non_dataframe_input(self):
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            group_and_aggregate(
                {"dept": ["Eng"], "salary": [100]},
                group_by=["dept"],
                aggregations={"salary": ["mean"]},
            )

    def test_reject_empty_group_by_list(self, clean_df):
        with pytest.raises(ValueError, match="'group_by' must contain at least one"):
            group_and_aggregate(clean_df, group_by=[], aggregations={"x": ["mean"]})

    def test_reject_nonexistent_grouping_column(self, clean_df):
        with pytest.raises(ValueError, match="Grouping column 'ghost' does not exist"):
            group_and_aggregate(clean_df, group_by=["ghost"], aggregations={"x": ["mean"]})

    def test_reject_empty_aggregations_dict(self, clean_df):
        with pytest.raises(ValueError, match="'aggregations' dictionary must contain"):
            group_and_aggregate(clean_df, group_by=["category"], aggregations={})

    def test_reject_nonexistent_aggregation_column(self, clean_df):
        with pytest.raises(ValueError, match="Aggregation column 'ghost' does not exist"):
            group_and_aggregate(
                clean_df, group_by=["category"], aggregations={"ghost": ["mean"]}
            )

    def test_reject_unsupported_aggregation_operation(self, clean_df):
        with pytest.raises(ValueError, match="Unsupported aggregation operation 'std'"):
            group_and_aggregate(
                clean_df, group_by=["category"], aggregations={"x": ["std"]}
            )

    def test_reject_invalid_parameter_types(self, clean_df):
        # group_by not a list
        with pytest.raises(TypeError, match="'group_by' must be a list"):
            group_and_aggregate(clean_df, group_by="category", aggregations={"x": ["mean"]})

        # group_by contains non-string
        with pytest.raises(TypeError, match="All column names in 'group_by' must be strings"):
            group_and_aggregate(clean_df, group_by=[123], aggregations={"x": ["mean"]})

        # aggregations not a dict
        with pytest.raises(TypeError, match="'aggregations' must be a dictionary"):
            group_and_aggregate(clean_df, group_by=["category"], aggregations=["mean"])

        # aggregations column has non-list operations
        with pytest.raises(TypeError, match="Operations for column 'x' must be a list"):
            group_and_aggregate(clean_df, group_by=["category"], aggregations={"x": "mean"})

        # aggregations operation is not a string
        with pytest.raises(TypeError, match="All operations for column 'x' must be strings"):
            group_and_aggregate(clean_df, group_by=["category"], aggregations={"x": [123]})

        # dropna not a boolean
        with pytest.raises(TypeError, match="'dropna' must be a boolean"):
            group_and_aggregate(
                clean_df, group_by=["category"], aggregations={"x": ["mean"]}, dropna="yes"
            )

    def test_input_dataframe_not_mutated(self, clean_df):
        df_copy = clean_df.copy()
        group_and_aggregate(clean_df, group_by=["category"], aggregations={"x": ["mean"]})
        assert clean_df.equals(df_copy)

    def test_results_contain_no_nan_or_inf(self):
        # When all target values in a group are NaN
        df = pd.DataFrame({
            "dept": ["Eng", "Sales"],
            "score": [float("nan"), 100.0],
        })
        result = group_and_aggregate(
            df, group_by=["dept"], aggregations={"score": ["mean"]}
        )
        eng = next(r for r in result["results"] if r["dept"] == "Eng")
        # Undefined mean should be converted to None, not float('nan')
        assert eng["score_mean"] is None

        # Verify no NaN or Inf in entire structure
        for record in result["results"]:
            for key, val in record.items():
                if isinstance(val, float):
                    assert not math.isnan(val)
                    assert not math.isinf(val)

    def test_column_names_containing_spaces_and_punctuation(self):
        df = pd.DataFrame({
            "dept name": ["Eng", "Sales", "Eng"],
            "unit price ($)": [10.5, 20.0, 30.5],
        })
        result = group_and_aggregate(
            df,
            group_by=["dept name"],
            aggregations={"unit price ($)": ["mean", "sum"]},
        )
        assert result["group_count"] == 2
        eng = next(r for r in result["results"] if r["dept name"] == "Eng")
        assert eng["unit price ($)_mean"] == 20.5
        assert eng["unit price ($)_sum"] == 41.0

    def test_output_format_is_json_serializable(self, employees_sample):
        result = group_and_aggregate(
            employees_sample,
            group_by=["department"],
            aggregations={"salary": ["mean", "max"], "bonus": ["count"]},
        )
        # Must serialize cleanly without errors
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

        # Round-trip verify
        deserialized = json.loads(serialized)
        assert deserialized["group_count"] == result["group_count"]
        assert deserialized["group_by"] == result["group_by"]

    def test_reject_output_name_collision(self):
        # Collision between group_by column name and output aggregation name
        df = pd.DataFrame({
            "salary_mean": ["A", "B"],
            "salary": [100.0, 200.0],
        })
        with pytest.raises(ValueError, match="Output column name collision"):
            group_and_aggregate(
                df, group_by=["salary_mean"], aggregations={"salary": ["mean"]}
            )

    def test_reject_duplicate_group_by_columns(self, clean_df):
        with pytest.raises(ValueError, match="duplicate column"):
            group_and_aggregate(
                clean_df, group_by=["category", "category"], aggregations={"x": ["mean"]}
            )

    def test_reject_duplicate_operations_on_same_column(self, clean_df):
        with pytest.raises(ValueError, match="Duplicate aggregation operation"):
            group_and_aggregate(
                clean_df, group_by=["category"], aggregations={"x": ["mean", "mean"]}
            )

    def test_reject_numeric_operations_on_non_numeric_columns(self):
        df = pd.DataFrame({
            "dept": ["Eng", "Sales"],
            "name": ["Alice", "Bob"],
        })
        with pytest.raises(ValueError, match="requires a numeric column"):
            group_and_aggregate(
                df, group_by=["dept"], aggregations={"name": ["mean"]}
            )

    def test_group_and_aggregate_on_employee_dataset(self, employee_df):
        result = group_and_aggregate(
            employee_df,
            group_by=["department"],
            aggregations={"salary": ["mean", "count"], "experience_years": ["max"]},
            dropna=True,
        )
        assert result["group_count"] > 0
        eng = next(r for r in result["results"] if r["department"] == "Engineering")
        assert eng["salary_mean"] > 0
        assert eng["salary_count"] == 8

    def test_group_and_aggregate_on_ecommerce_dataset(self, ecommerce_df):
        result = group_and_aggregate(
            ecommerce_df,
            group_by=["category", "region"],
            aggregations={"quantity": ["sum"], "unit_price": ["mean"]},
            dropna=True,
        )
        assert result["group_count"] > 0
        assert "quantity_sum" in result["results"][0]
        assert "unit_price_mean" in result["results"][0]


# ── 7. Ground Truth 20x6 Analytical Engine End-to-End Verification ─────────────

def test_ground_truth_20x6_full_analytical_suite():
    """Verify all six analytical tools against ground_truth_20x6.csv as specified in Step 3."""
    csv_path = Path(__file__).resolve().parent.parent / "ground_truth_20x6.csv"
    assert csv_path.is_file(), f"ground_truth_20x6.csv must exist at {csv_path}"
    df = pd.read_csv(csv_path)

    # 1. get_dataset_overview: 20 rows, 6 columns, zero missing values
    ov = get_dataset_overview(df)
    assert ov["rows"] == 20
    assert ov["columns"] == 6
    assert sum(ov["missing_counts"].values()) == 0

    # 2. get_summary_statistics:
    # Mean experience_years = 10.5, Mean salary = 2050, Min salary = 1100, Max salary = 3000
    stats = get_summary_statistics(df)
    assert stats["numeric"]["experience_years"]["mean"] == 10.5
    assert stats["numeric"]["salary"]["mean"] == 2050.0
    assert stats["numeric"]["salary"]["min"] == 1100.0
    assert stats["numeric"]["salary"]["max"] == 3000.0

    # 3. analyze_correlations:
    # Pearson correlation between experience_years and salary = 1.0
    corr = analyze_correlations(df, columns=["experience_years", "salary"])
    r_val = corr["matrix"]["experience_years"]["salary"]
    assert round(r_val, 4) == 1.0

    # 4. filter_data:
    # Filtering experience_years >= 18 returns E18, E19, E20
    filt = filter_data(df, [{"column": "experience_years", "operator": "gte", "value": 18}])
    assert list(filt["employee"]) == ["E18", "E19", "E20"]

    # 5. group_and_aggregate:
    # Grouping by department gives 10 employees per department.
    # Alpha average salary = 1550, Beta average salary = 2550
    grp = group_and_aggregate(
        df,
        group_by=["department"],
        aggregations={"employee": ["count"], "salary": ["mean"]},
    )
    by_dept = {r["department"]: r for r in grp["results"]}
    assert by_dept["Alpha"]["employee_count"] == 10
    assert by_dept["Alpha"]["salary_mean"] == 1550.0
    assert by_dept["Beta"]["employee_count"] == 10
    assert by_dept["Beta"]["salary_mean"] == 2550.0

    # 6. generate_visualization:
    # Scatter visualization represents all 20 observations
    fig = generate_visualization(df, chart_type="scatter", x_column="experience_years", y_column="salary")
    total_pts = sum(len(trace.x) for trace in fig.data)
    assert total_pts == 20


