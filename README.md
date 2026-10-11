# neo-applied-ml

**A beginner-friendly, technically sound Applied ML project** built for BYTE's Basic Applied ML task.

The project is developed in phases:
- **Phase 1 (current)** — Analytical engine: five data-analysis functions, a sample dataset, and a full test suite.
- Phase 2 *(planned)* — LLM agent that calls Phase 1 functions in response to natural-language questions.
- Phase 3 *(planned)* — Streamlit UI.

---

## Project structure

```
neo-applied-ml/
├── .gitignore            # keeps secrets, venvs, and caches out of Git
├── README.md             # this file
├── PROGRESS.md           # what's been built and what's next
├── requirements.txt      # Python dependencies
├── pyproject.toml        # pytest configuration
│
├── app/
│   ├── __init__.py
│   └── analysis/
│       ├── __init__.py        # re-exports all five functions
│       ├── overview.py        # get_dataset_overview
│       ├── statistics.py      # get_summary_statistics
│       ├── filtering.py       # filter_data
│       ├── correlations.py    # analyze_correlations
│       └── visualization.py   # generate_visualization
│
├── examples/
│   ├── sample_data.csv   # 20-row synthetic student dataset
│   └── demo.py           # runnable demonstration of all five functions
│
└── tests/
    ├── conftest.py       # shared pytest fixtures
    └── test_analysis.py  # 72 unit tests
```

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd neo-applied-ml

# 2. Create a virtual environment (keeps dependencies isolated)
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Running the demo

```bash
python -m examples.demo
```

This prints a summary of all five analytical functions applied to the sample dataset,
and saves an interactive Plotly scatter plot as `examples/scatter_demo.html`.

---

## Running the tests

```bash
pytest tests/ -v
```

Expected output: **72 passed**.

---

## The five analytical functions

All functions live in `app/analysis/` and are importable from the package:

```python
from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
)
```

| Function | What it does | Returns |
|---|---|---|
| `get_dataset_overview(df)` | Shape, dtypes, missing counts, duplicates | `dict` |
| `get_summary_statistics(df, top_n)` | Mean/median/std for numeric; top-N value counts for categoricals | `dict` |
| `filter_data(df, conditions)` | Filter rows by one or more column conditions | `pd.DataFrame` |
| `analyze_correlations(df, columns, top_n)` | Pearson correlation matrix + top pairs | `dict` |
| `generate_visualization(df, chart_type, ...)` | bar / line / scatter / histogram / box chart | `plotly.Figure` |

### `filter_data` — supported operators

| Operator | Meaning | Example |
|---|---|---|
| `eq` | equal to | `{"column": "city", "operator": "eq", "value": "London"}` |
| `ne` | not equal to | `{"column": "grade", "operator": "ne", "value": "D"}` |
| `gt` | greater than | `{"column": "score", "operator": "gt", "value": 80}` |
| `gte` | ≥ | `{"column": "score", "operator": "gte", "value": 80}` |
| `lt` | less than | `{"column": "age", "operator": "lt", "value": 22}` |
| `lte` | ≤ | `{"column": "age", "operator": "lte", "value": 22}` |
| `contains` | substring match (case-insensitive) | `{"column": "city", "operator": "contains", "value": "lon"}` |
| `isnull` | value is missing | `{"column": "grade", "operator": "isnull"}` |
| `notnull` | value is present | `{"column": "score", "operator": "notnull"}` |

Multiple conditions are combined with **AND** (all must be true).

---

## Known limitations (Phase 1)

- `filter_data` combines conditions with AND only; OR logic is not yet supported.
- `analyze_correlations` uses Pearson (linear) correlation only; Spearman / Kendall are not available.
- `generate_visualization` does not support pie charts, heatmaps, or multi-panel layouts.
- No support for datetime columns in filtering or visualisation beyond treating them as strings.
- The sample dataset is small (20 rows) and synthetic; real-world datasets will surface additional edge cases.
