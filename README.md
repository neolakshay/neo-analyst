# neo-applied-ml

**A beginner-friendly, technically sound Applied ML project** built for BYTE's Basic Applied ML task.

The project is developed in phases:
- **Phase 1** — Analytical engine: six data-analysis functions, validation datasets, and a full test suite.
- **Phase 2 (Step 1 complete)** — Core LLM-powered data analysis agent that selects and executes Phase 1 tools via the official OpenAI tool-calling interface.
- **Phase 3** *(planned)* — Streamlit UI.

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
│   ├── analysis/              # Phase 1: Analytical Engine
│   │   ├── __init__.py        # re-exports all six functions
│   │   ├── overview.py        # get_dataset_overview
│   │   ├── statistics.py      # get_summary_statistics
│   │   ├── filtering.py       # filter_data
│   │   ├── correlations.py    # analyze_correlations
│   │   ├── visualization.py   # generate_visualization
│   │   └── aggregation.py     # group_and_aggregate
│   │
│   └── agent/                 # Phase 2: LLM Analysis Agent
│       ├── __init__.py        # AnalysisAgent, AgentResult, ToolCallRecord
│       ├── tools.py           # Tool schemas & local execution dispatcher
│       └── agent.py           # Multi-round tool-calling loop & orchestration
│
├── examples/
│   ├── sample_data.csv        # 20-row student dataset
│   ├── ecommerce_sales.csv    # 150-row e-commerce sales dataset
│   ├── employee_analytics.csv # 50-row employee analytics dataset
│   ├── edge_cases.csv         # 20-row edge cases dataset
│   ├── demo.py                # demonstration of all six analytical functions
│   └── agent_demo.py          # interactive terminal demo of the LLM agent
│
└── tests/
    ├── conftest.py            # shared pytest fixtures
    ├── test_analysis.py       # 112 analytical engine unit tests
    └── test_agent.py          # 23 agent unit tests (mocked OpenAI client)
```

---

## Configuration

The agent uses the official OpenAI Python SDK. Create a `.env` file in the project root (this file is excluded from git):

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini  # optional (default: gpt-4o-mini)
```

Environment variables are loaded automatically using `python-dotenv`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/neolakshay/neo-applied-ml
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

## Running the demos

### 1. Analytical Engine Demo
```bash
python -m examples.demo
```
This prints a summary of all six analytical functions applied to the sample dataset,
and saves an interactive Plotly scatter plot as `examples/scatter_demo.html`.

### 2. LLM Analysis Agent Demo
```bash
python -m examples.agent_demo "Which department has the highest average salary?"
```
Or run interactively without arguments to input questions from the terminal. Generated Plotly charts are saved as HTML files in `examples/`.

---

## Running the tests

```bash
pytest tests/ -v
```

Expected output: **135 passed** (112 analytical engine tests + 23 agent tests).

---

## The six analytical functions

All functions live in `app/analysis/` and are importable from the package:

```python
from app.analysis import (
    get_dataset_overview,
    get_summary_statistics,
    filter_data,
    analyze_correlations,
    generate_visualization,
    group_and_aggregate,
)
```

| Function | What it does | Returns |
|---|---|---|
| `get_dataset_overview(df)` | Shape, dtypes, missing counts, duplicates | `dict` |
| `get_summary_statistics(df, top_n)` | Mean/median/std for numeric; top-N value counts for categoricals | `dict` |
| `filter_data(df, conditions)` | Filter rows by one or more column conditions | `pd.DataFrame` |
| `analyze_correlations(df, columns, top_n)` | Pearson correlation matrix + top pairs | `dict` |
| `generate_visualization(df, chart_type, ...)` | bar / line / scatter / histogram / box chart | `plotly.Figure` |
| `group_and_aggregate(df, group_by, aggregations, dropna)` | Group by 1+ columns and compute summary statistics | `dict` |

### `group_and_aggregate` — grouped summary statistics

Groups tabular data using pandas groupby and calculates summary statistics per group.

Supported operations:
- `mean` — arithmetic average of non-null values (numeric only)
- `median` — middle non-null value (numeric only)
- `sum` — sum of non-null values (numeric only)
- `min` — minimum non-null value
- `max` — maximum non-null value
- `count` — non-null observation count for the target column

#### Example call:

```python
result = group_and_aggregate(
    df,
    group_by=["department"],
    aggregations={
        "salary": ["mean", "median", "max"],
        "employee_id": ["count"],
    },
    dropna=True,
)
```

#### Return format:

```json
{
  "group_by": ["department"],
  "aggregations": {
    "salary": ["mean", "median", "max"],
    "employee_id": ["count"]
  },
  "group_count": 2,
  "results": [
    {
      "department": "Engineering",
      "salary_mean": 1250000.0,
      "salary_median": 1200000.0,
      "salary_max": 1800000.0,
      "employee_id_count": 8
    },
    {
      "department": "Sales",
      "salary_mean": 950000.0,
      "salary_median": 900000.0,
      "salary_max": 1400000.0,
      "employee_id_count": 6
    }
  ]
}
```

The output contains standard Python types (`dict`, `list`, `str`, `int`, `float`, `None`) and is fully JSON-serializable.

---

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
- `group_and_aggregate` does not support custom lambda expressions or window functions (limited to the 6 core aggregation operations).
