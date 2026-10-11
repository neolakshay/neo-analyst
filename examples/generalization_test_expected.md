# Expected Analytical Results: Generalization Test Dataset

This document provides independently calculated ground-truth expectations for `examples/generalization_test.csv`. These expectations were computed directly from the CSV using standard `pandas` and Python mathematical libraries without invoking any analytical or agent tools from `neo-applied-ml`.

---

## 1. Dataset Dimensions and Exact Column Order

- **Rows:** 15 data rows (excluding header)
- **Columns:** 6 columns
- **Exact Column Order:**
  1. `employee_id`
  2. `team`
  3. `years_experience`
  4. `monthly_salary`
  5. `satisfaction_score`
  6. `weekly_hours`

---

## 2. Data Types (as Loaded by pandas)

When loaded via `pd.read_csv('examples/generalization_test.csv')`:

| Column | pandas dtype | Python Type Representation | Notes |
|---|---|---|---|
| `employee_id` | `object` / `string` | `str` | Alphanumeric unique IDs (`EMP01` to `EMP15`) |
| `team` | `object` / `string` | `str` or `NaN` | Categorical text with 1 missing entry |
| `years_experience` | `int64` | `int` | Integer numeric (no missing values) |
| `monthly_salary` | `float64` | `float` or `NaN` | Numeric (cast to `float64` due to 1 missing value) |
| `satisfaction_score` | `float64` | `float` or `NaN` | Decimal numeric (1 decimal place, 1 missing value) |
| `weekly_hours` | `int64` | `int` | Integer numeric (no missing values) |

---

## 3. Missing-Value Counts

| Column | Missing Count | Non-Missing Count | Percentage Missing |
|---|---|---|---|
| `employee_id` | 0 | 15 | 0.0% |
| `team` | 1 | 14 | 6.67% (Row `EMP08`) |
| `years_experience` | 0 | 15 | 0.0% |
| `monthly_salary` | 1 | 14 | 6.67% (Row `EMP05`) |
| `satisfaction_score` | 1 | 14 | 6.67% (Row `EMP10`) |
| `weekly_hours` | 0 | 15 | 0.0% |
| **Total Missing Cells** | **3** | **87** | **3.33%** |

---

## 4. Duplicate Records

- **Duplicate Complete Rows:** 0 (all 15 rows are unique).
- **Unique `employee_id` Count:** 15 out of 15 (100% unique primary identifiers).

---

## 5. Summary Statistics for Numeric Columns

All statistics follow standard sample statistical formulas with Bessel's correction (`ddof=1` for sample standard deviation):

| Statistic | `years_experience` | `monthly_salary` | `satisfaction_score` | `weekly_hours` |
|---|---|---|---|---|
| **Non-missing Count** | 15 | 14 | 14 | 15 |
| **Missing Count** | 0 | 1 | 1 | 0 |
| **Mean** | `5.000000` | `3300.000000` | `7.000000` | `40.000000` |
| **Median** | `5.000000` | `3200.000000` | `7.000000` | `39.000000` |
| **Minimum** | `1.000000` | `2400.000000` | `5.600000` | `35.000000` |
| **Maximum** | `9.000000` | `4300.000000` | `8.000000` | `47.000000` |
| **Sample Std Dev (`ddof=1`)** | `2.618615` | `648.074070` | `0.698350` | `3.664502` |

---

## 6. Pearson Correlations (Pairwise Complete Observations)

Correlations are computed pairwise using only observations where both variables are non-null:

| Variable Pair | Observations Used | Pearson Correlation ($r$) | Direction | Interpretation |
|---|---|---|---|---|
| `years_experience` vs `monthly_salary` | 14 | `+0.987135` | Positive | Strong linear salary progression with experience |
| `years_experience` vs `weekly_hours` | 15 | `+0.982561` | Positive | Senior staff work slightly longer hours |
| `monthly_salary` vs `weekly_hours` | 14 | `+0.982869` | Positive | Strong positive association |
| `years_experience` vs `satisfaction_score` | 14 | `-0.966026` | Negative | Strong negative linear correlation |
| `monthly_salary` vs `satisfaction_score` | 13 | `-0.976882` | Negative | Higher salary associated with lower satisfaction |
| `satisfaction_score` vs `weekly_hours` | 14 | `-0.947755` | Negative | Longer hours correlate with decreased satisfaction |

### Full Correlation Matrix:

| | `years_experience` | `monthly_salary` | `satisfaction_score` | `weekly_hours` |
|---|---|---|---|---|
| **`years_experience`** | `1.000000` | `0.987135` | `-0.966026` | `0.982561` |
| **`monthly_salary`** | `0.987135` | `1.000000` | `-0.976882` | `0.982869` |
| **`satisfaction_score`** | `-0.966026` | `-0.976882` | `1.000000` | `-0.947755` |
| **`weekly_hours`** | `0.982561` | `0.982869` | `-0.947755` | `1.000000` |

---

## 7. Filtering Results

### Filter A: `years_experience >= 5 AND weekly_hours < 40`
- **Total Matches:** 2 records
- **Matching Records:**
  1. `EMP05` (Team: `Core`, `years_experience`: 5, `weekly_hours`: 39, `monthly_salary`: `NaN`, `satisfaction_score`: 6.5)
  2. `EMP12` (Team: `Platform`, `years_experience`: 5, `weekly_hours`: 39, `monthly_salary`: 3300.0, `satisfaction_score`: 7.0)

> **Missing Value Handling:** `EMP05` has a missing `monthly_salary`, but satisfies both filter predicates (`5 >= 5` and `39 < 40`). Missing values in unqueried columns do not invalidate matches.

---

### Filter B: `monthly_salary > 3000`
- **Total Matches:** 8 records
- **Matching Records:**
  1. `EMP04` (Team: `Core`, `monthly_salary`: 3100.0)
  2. `EMP08` (Team: `NaN`, `monthly_salary`: 3800.0)
  3. `EMP09` (Team: `Growth`, `monthly_salary`: 3800.0)
  4. `EMP10` (Team: `Growth`, `monthly_salary`: 4200.0)
  5. `EMP12` (Team: `Platform`, `monthly_salary`: 3300.0)
  6. `EMP13` (Team: `Platform`, `monthly_salary`: 3600.0)
  7. `EMP14` (Team: `Platform`, `monthly_salary`: 4000.0)
  8. `EMP15` (Team: `Platform`, `monthly_salary`: 4300.0)

> **Missing Value Handling:** `EMP05` has `monthly_salary` = `NaN`. In standard relational filtering, `NaN > 3000` evaluates to `False`, so `EMP05` is correctly excluded. `EMP07` with `monthly_salary` = 3000.0 is excluded because the condition is strictly greater than (`>`).

---

## 8. Grouping and Aggregation by Team

### Primary Behavior (Standard pandas `dropna=True`):
When grouping on `team` with missing keys dropped:

| `team` | Total Rows | Non-Missing Salaries | Sum of Salaries | Mean Monthly Salary |
|---|---|---|---|---|
| **`Core`** | 5 | 4 | 10,800.0 | **`2,700.0`** |
| **`Growth`** | 4 | 4 | 13,600.0 | **`3,400.0`** |
| **`Platform`** | 5 | 5 | 18,000.0 | **`3,600.0`** |

- **Group Count:** 3 groups
- **Highest Mean Salary Team:** `Platform` (`3,600.0`)
- **Lowest Mean Salary Team:** `Core` (`2,700.0`)

---

### Unassigned / Missing Team Accounting (`dropna=False`):
If the grouping operation preserves unassigned categories:

| `team` | Total Rows | Non-Missing Salaries | Sum of Salaries | Mean Monthly Salary |
|---|---|---|---|---|
| **`Core`** | 5 | 4 | 10,800.0 | `2,700.0` |
| **`Growth`** | 4 | 4 | 13,600.0 | `3,400.0` |
| **`Platform`** | 5 | 5 | 18,000.0 | `3,600.0` |
| **`NaN` (Unassigned)** | 1 (`EMP08`) | 1 | 3,800.0 | `3,800.0` |

---

## 9. Non-Missing Salary Counts Per Team

- `Core`: **4** non-missing salaries (`EMP01`, `EMP02`, `EMP03`, `EMP04`; `EMP05` is missing)
- `Growth`: **4** non-missing salaries (`EMP06`, `EMP07`, `EMP09`, `EMP10`)
- `Platform`: **5** non-missing salaries (`EMP11`, `EMP12`, `EMP13`, `EMP14`, `EMP15`)
- `NaN` (Unassigned): **1** non-missing salary (`EMP08`)
- **Overall Total Non-Missing Salaries:** 14

---

## 10. Scatter Plot Expectations: `years_experience` vs `monthly_salary`

- **X-axis Variable:** `years_experience`
- **Y-axis Variable:** `monthly_salary`
- **Total Dataset Rows:** 15
- **Usable Observations (Both coordinates non-missing):** **14**
- **Excluded Rows:** Exactly 1 row (`EMP05`), because `monthly_salary` is `NaN`.
- **Expected Visual Trend:** Linear upward sloping scatter with strong correlation ($r \approx 0.987$).

---

## 11. Impact of Missing Values on Analytical Operations

1. **Univariate Statistics (Mean, Std Dev):**
   - pandas skips `NaN` values by default (`skipna=True`).
   - For `monthly_salary` and `satisfaction_score`, the denominator is 14 (not 15).
2. **Correlation Calculation:**
   - Pairwise deletion applies: only rows with valid pairs are included.
   - For `monthly_salary` vs `satisfaction_score`, both `EMP05` (missing salary) and `EMP10` (missing satisfaction) are omitted, leaving 13 complete pairs.
3. **Filtering:**
   - Comparisons against `NaN` evaluate to `False` (e.g. `NaN > 3000` is `False`). Missing values are never silently coerced to zero.
4. **Grouping:**
   - Standard pandas `groupby('team')` omits `EMP08` (missing team) unless explicitly specified with `dropna=False`.
   - Within group `Core`, the mean salary divides by 4 valid observations, yielding an exact average of `2,700.0`.

---

## 12. pandas Dtype and Aggregation Details

- **Type Promotion:** Because pandas integer columns cannot represent standard `NaN` without nullable integer types (`Int64`), the presence of `NaN` in `monthly_salary` automatically promotes it from integer to `float64`.
- **Sample vs Population Statistics:** pandas `.std()` computes the sample standard deviation with $N-1$ degrees of freedom (`ddof=1`), whereas NumPy's default `.std()` uses $N$ (`ddof=0`). The sample values documented here align with pandas default behavior.
- **Aggregation Column Naming:** When aggregating via pandas groupby, calculating counts of non-missing values (`monthly_salary.count()`) yields 4 for `Core`, whereas row count (`employee_id.count()`) yields 5.
