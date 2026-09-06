# Power BI Dashboard — Build Instructions
## Smart Mobile Data Usage Prediction & Plan Recommendation System

Power BI Desktop cannot be generated headlessly here, so this folder ships
the **Power BI-ready CSV data model** plus exact steps + DAX to build
`mobile_data_usage_dashboard.pbix` yourself in Power BI Desktop.

## 1. Files in this folder (import all five as separate tables)

| File | Role | Rows |
|---|---|---|
| `fact_daily_usage.csv` | Fact table — one row per user per day | 90,000 |
| `dim_customers.csv` | Customer dimension — one row per user | 500 |
| `dim_date.csv` | Date dimension | 180 |
| `fact_predictions.csv` | Prediction/risk fact table — one row per user (latest snapshot) | 500 |
| `fact_whatif.csv` | What-if (−20% streaming) scenario table — one row per user | 500 |

## 2. Data model (Model view)

Create these relationships (all **1-to-many**, single direction, from the "1" dimension to the "many" fact):

- `dim_customers[User_ID]` → `fact_daily_usage[User_ID]`
- `dim_customers[User_ID]` → `fact_predictions[User_ID]`
- `dim_customers[User_ID]` → `fact_whatif[User_ID]`
- `dim_date[Date]` → `fact_daily_usage[Date]`

Keep it a simple star schema: one customer dimension, one date dimension,
three fact tables. No snowflaking or bridge tables are needed — `Plan_GB`,
`Occupation`, `Age`, and `Network_Type` all live once in `dim_customers` and
are pulled into every visual through the relationships above.

## 3. DAX measures (create in a new blank "Measures" table, or on `fact_daily_usage`)

```dax
Total Users = DISTINCTCOUNT(dim_customers[User_ID])

Avg Daily Usage MB = AVERAGE(fact_daily_usage[Mobile_Data_MB])

Avg Monthly Usage MB =
DIVIDE(
    SUM(fact_daily_usage[Mobile_Data_MB]),
    DISTINCTCOUNT(fact_daily_usage[User_ID]) * (DISTINCTCOUNT(fact_daily_usage[Date]) / 6)
)
-- (6 = number of calendar months spanned by the 180-day dataset)

High Risk Users = CALCULATE(DISTINCTCOUNT(fact_predictions[User_ID]), fact_predictions[Risk_Level] = "High")

Avg Plan Utilization % =
AVERAGEX(
    fact_predictions,
    DIVIDE(fact_predictions[Predicted_Month_End_Usage_MB], fact_predictions[Plan_GB] * 1024)
) * 100
```

### What-if parameter (Page 4)

1. Modeling ribbon → **New Parameter → Numeric range**: name it
   `Streaming Reduction %`, Min 0, Max 50, Increment 5, Default 20.
   This creates a `Streaming Reduction %` table + `Streaming Reduction % Value` measure automatically.
2. Add these measures next to it:

```dax
Adjusted Predicted Usage MB =
VAR ReductionPct = [Streaming Reduction % Value] / 100
VAR StreamShare = AVERAGE(fact_whatif[Streaming_Share_Of_Usage])
RETURN
    AVERAGE(fact_predictions[Predicted_Month_End_Usage_MB]) * (1 - StreamShare * ReductionPct)

Data Saved MB (Parameterized) =
[Avg Plan Utilization %] -- placeholder base
VAR Base = AVERAGE(fact_predictions[Predicted_Month_End_Usage_MB])
VAR Adjusted = [Adjusted Predicted Usage MB]
RETURN Base - Adjusted

Risk After (Parameterized) =
VAR PlanMB = AVERAGE(fact_predictions[Plan_GB]) * 1024
VAR Pct = DIVIDE([Adjusted Predicted Usage MB], PlanMB)
RETURN
    SWITCH(
        TRUE(),
        Pct <= 0.85, "Low",
        Pct <= 1.00, "Medium",
        "High"
    )
```

The pre-computed `fact_whatif.csv` (fixed at a 20% reduction, matching the
Python what-if analysis exactly) drives Page 4's static comparison; the DAX
parameter above lets the user interactively slide the reduction percentage
for exploration on top of that baseline.

## 4. Pages

### Page 1 — Executive Overview
- **Cards**: `Total Users`, `Avg Daily Usage MB`, `Avg Monthly Usage MB`, `High Risk Users`, `Avg Plan Utilization %`
- **Line chart**: `dim_date[Date]` (axis) vs `Avg Daily Usage MB` (value) — daily usage trend
- **Bar chart**: `dim_customers[Occupation]` vs `Avg Daily Usage MB`
- **Bar chart**: app/category columns from `fact_daily_usage` (YouTube_MB, Netflix_MB, Instagram_MB, WhatsApp_MB, Gaming_MB, Browsing_MB, Work_Study_MB) — use "Transform Data" to unpivot these into an `App/Metric` column first, then bar-chart the unpivoted average
- **Clustered column**: `dim_date[IsWeekend]` vs `Avg Daily Usage MB` (weekend vs weekday)
- **Bar chart**: `dim_customers[Network_Type]` vs `Avg Daily Usage MB`

### Page 2 — Prediction & Risk
Table/matrix visual bound to `fact_predictions`, columns:
`User_ID, Current_Usage_MB, Predicted_Next_Day_Usage_MB, Predicted_Month_End_Usage_MB, Plan_GB, Remaining_Plan_MB, Expected_Shortage_MB, Risk_Level, Expected_Exhaustion_Date`
Add a **donut chart** of `Risk_Level` counts, and a **stacked bar** of `Risk_Level` by `Occupation` (via relationship to `dim_customers`).

### Page 3 — Customer Details
- Add a **slicer** on `dim_customers[User_ID]`.
- **Card visuals** for: Occupation, Plan_GB, Plan_Price, Current_Usage_MB, Remaining_Plan_MB, `Avg Daily Usage MB`, Predicted_Next_Day_Usage_MB, Predicted_Month_End_Usage_MB, Risk_Level, Expected_Exhaustion_Date, Recommendation (all pulled from `fact_predictions` + `dim_customers` filtered by the slicer).
- **Line chart** of that user's daily usage over time from `fact_daily_usage`.

### Page 4 — What-If Analysis
- Add the `Streaming Reduction %` parameter slicer.
- **KPI cards**: `Avg Plan Utilization %` (current), `Adjusted Predicted Usage MB`, `Data Saved MB (Parameterized)`.
- **Clustered bar**: Original_Risk vs New_Risk counts from `fact_whatif.csv` (add a "count" measure grouped by each column, or use a matrix visual with `Original_Risk` on rows and `New_Risk` on columns to show the risk-transition matrix directly).

## 5. Save

File → Save As → `mobile_data_usage_dashboard.pbix`.
