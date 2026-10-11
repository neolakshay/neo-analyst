# examples/demo.py
"""
Quick demonstration of all five analytical functions.

Run from the project root:
    python -m examples.demo

Or, if you're inside the examples/ directory:
    python demo.py  (won't work unless you adjust sys.path)
"""

import pathlib
import pandas as pd

# ── Make sure the project root is on the path ─────────────────────────────────
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
    group_and_aggregate,
)

# ── Load the sample dataset ───────────────────────────────────────────────────
DATA_PATH = pathlib.Path(__file__).parent / "sample_data.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DEMO: neo-applied-ml Analytical Engine")
print("=" * 60)

# 1. Dataset overview
print("\n--- 1. Dataset Overview ---")
overview = get_dataset_overview(df)
print(f"  Shape      : {overview['rows']} rows × {overview['columns']} columns")
print(f"  Columns    : {overview['column_names']}")
print(f"  Duplicates : {overview['duplicate_rows']}")
print("  Missing values per column:")
for col, count in overview["missing_counts"].items():
    if count > 0:
        print(f"    {col}: {count} ({overview['missing_pct'][col]}%)")

# 2. Summary statistics
print("\n--- 2. Summary Statistics ---")
stats = get_summary_statistics(df, top_n=3)
print("  Numeric columns:")
for col, s in stats["numeric"].items():
    print(f"    {col}: mean={s['mean']}, median={s['median']}, std={s['std']}")
print("  Categorical columns (top 3 values):")
for col, counts in stats["categorical"].items():
    print(f"    {col}: {counts}")

# 3. Filter data
print("\n--- 3. Filter Data ---")
high_scorers = filter_data(df, [
    {"column": "score", "operator": "gte", "value": 80},
    {"column": "passed", "operator": "eq",  "value": True},
])
print(f"  Students with score ≥ 80 and passed=True: {len(high_scorers)} rows")
print(high_scorers[["name", "score", "grade"]].to_string(index=False))

# 4. Correlations
print("\n--- 4. Analyze Correlations ---")
corr = analyze_correlations(df, top_n=3)
print("  Top positive pairs:")
for col_a, col_b, r in corr["top_positive"]:
    print(f"    {col_a} ↔ {col_b}: r = {r}")
print("  Top negative pairs:")
for col_a, col_b, r in corr["top_negative"]:
    print(f"    {col_a} ↔ {col_b}: r = {r}")

# 5. Visualization (saves to HTML file instead of opening browser)
print("\n--- 5. Generate Visualization ---")
fig = generate_visualization(
    df.dropna(subset=["score", "study_hours"]),
    chart_type="scatter",
    x_column="study_hours",
    y_column="score",
    color_column="grade",
    title="Study Hours vs Score",
)
out_path = pathlib.Path(__file__).parent / "scatter_demo.html"
fig.write_html(str(out_path))
# 6. Group and aggregate
print("\n--- 6. Group and Aggregate ---")
grouped_summary = group_and_aggregate(
    df,
    group_by=["grade"],
    aggregations={
        "score": ["mean", "max"],
        "student_id": ["count"],
    },
    dropna=True,
)
print(f"  Grouped by grade ({grouped_summary['group_count']} groups):")
for grp in grouped_summary["results"]:
    print(f"    Grade {grp['grade']}: count={grp['student_id_count']}, mean_score={grp['score_mean']}, max_score={grp['score_max']}")

print("\nDemo complete!")
