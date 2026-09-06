"""
Step 4: Prediction Output, 7-Day Recursive Forecast, Month-End Forecast,
Risk Classification, Exhaustion Date, Recommendation Engine, What-If Analysis.

All numbers here come from the trained model (loaded from disk) and the
actual cleaned dataset — nothing is hard-coded or invented.

Output: data/output/mobile_usage_predictions.csv
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_PATH = BASE_DIR / "data" / "cleaned" / "mobile_data_usage_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "best_next_day_usage_model.joblib"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS_NUMERIC = [
    "Lag1_Mobile_MB", "RollMean3_Mobile_MB", "RollMean7_Mobile_MB",
    "RollMean14_Mobile_MB", "RollMean30_Mobile_MB", "Lag1_Streaming_MB",
    "Lag1_Screen_Time_Hours", "Usage_Growth_Ratio", "Age", "Plan_GB",
    "WiFi_Availability_Lag1",
]
FEATURE_COLS_BINARY = ["Is_Weekend", "Is_Holiday"]
FEATURE_COLS_CATEGORICAL = ["Occupation", "Network_Type"]
ALL_FEATURES = FEATURE_COLS_NUMERIC + FEATURE_COLS_BINARY + FEATURE_COLS_CATEGORICAL

# ---------------------------------------------------------------------------
# Business-rule configuration (explainable, adjustable thresholds)
# ---------------------------------------------------------------------------
# Risk thresholds are based on *predicted month-end usage as a % of the plan*:
#   - LOW: predicted to land at or below 85% of the plan -> comfortable buffer
#   - MEDIUM: predicted between 85% and 100% -> approaching the limit
#   - HIGH: predicted to exceed 100% of the plan -> expected to run out
# These are configurable business thresholds, not model outputs.
RISK_LOW_MAX_PCT = 0.85
RISK_MEDIUM_MAX_PCT = 1.00

FORECAST_HORIZON_DAYS = 7
STREAMING_REDUCTION_PCT = 0.20  # what-if scenario

model = joblib.load(MODEL_PATH)

df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 1. Rebuild the SAME leakage-safe features used at training time, for every
#    row (we need each user's most recent day as the forecast anchor).
# ---------------------------------------------------------------------------
g = df.groupby("User_ID")
shifted_usage = g["Mobile_Data_MB"].shift(1)
df["Lag1_Mobile_MB"] = shifted_usage
for window in [3, 7, 14, 30]:
    df[f"RollMean{window}_Mobile_MB"] = (
        shifted_usage.groupby(df["User_ID"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
streaming_mb = df["YouTube_MB"] + df["Netflix_MB"]
df["Lag1_Streaming_MB"] = streaming_mb.groupby(df["User_ID"]).shift(1)
df["Lag1_Screen_Time_Hours"] = g["Screen_Time_Hours"].shift(1)
roll3, roll7 = df["RollMean3_Mobile_MB"], df["RollMean7_Mobile_MB"]
df["Usage_Growth_Ratio"] = (roll3 / roll7.replace(0, np.nan)).fillna(1.0)
df["WiFi_Availability_Lag1"] = g["WiFi_Availability"].shift(1)

# ---------------------------------------------------------------------------
# 2. Anchor each user at their LAST available day in the dataset — this is
#    "today" from the model's point of view; we forecast forward from here.
# ---------------------------------------------------------------------------
last_rows = df.loc[df.groupby("User_ID")["Date"].idxmax()].copy()
last_rows = last_rows.dropna(subset=FEATURE_COLS_NUMERIC)  # need full lag history to forecast
print(f"Users with enough history to forecast: {len(last_rows)} / {df['User_ID'].nunique()}")

anchor_date = last_rows["Date"].max()
print("Anchor (latest) date in dataset:", anchor_date.date())

# ---------------------------------------------------------------------------
# 3. Current-month-to-date actual usage (calendar month of the anchor date)
# ---------------------------------------------------------------------------
df["Year_Month"] = df["Date"].dt.to_period("M")
current_period = anchor_date.to_period("M")
month_days_total = current_period.days_in_month
day_of_month = anchor_date.day

mtd_actual = (
    df[df["Year_Month"] == current_period]
    .groupby("User_ID")["Mobile_Data_MB"].sum()
    .rename("Current_Month_Usage_MB")
)

# ---------------------------------------------------------------------------
# 4. Recursive 7-day forecast per user (Day T+1 prediction feeds Day T+2's
#    lag features, etc.). We NEVER use future actual values here.
# ---------------------------------------------------------------------------
def recursive_forecast(row, horizon=FORECAST_HORIZON_DAYS):
    """Roll a user's state forward `horizon` days using only model outputs
    and calendar facts (weekend/holiday flags for the forecast dates)."""
    state = row.copy()
    # maintain the actual recent-history window (most recent up to 30 real days)
    hist = list(recent_history.get(row["User_ID"], []))  # oldest -> newest actual Mobile_Data_MB
    preds = []
    cur_date = row["Date"]
    for step in range(1, horizon + 1):
        next_date = cur_date + pd.Timedelta(days=step)
        feat = {
            "Lag1_Mobile_MB": hist[-1] if hist else state["Lag1_Mobile_MB"],
            "RollMean3_Mobile_MB": np.mean(hist[-3:]) if hist else state["RollMean3_Mobile_MB"],
            "RollMean7_Mobile_MB": np.mean(hist[-7:]) if hist else state["RollMean7_Mobile_MB"],
            "RollMean14_Mobile_MB": np.mean(hist[-14:]) if hist else state["RollMean14_Mobile_MB"],
            "RollMean30_Mobile_MB": np.mean(hist[-30:]) if hist else state["RollMean30_Mobile_MB"],
            "Lag1_Streaming_MB": state["Lag1_Streaming_MB"],  # last known streaming split (kept static; app breakdown is not recursively forecast)
            "Lag1_Screen_Time_Hours": state["Lag1_Screen_Time_Hours"],
            "Usage_Growth_Ratio": (np.mean(hist[-3:]) / np.mean(hist[-7:])) if hist and np.mean(hist[-7:]) > 0 else 1.0,
            "Age": state["Age"],
            "Plan_GB": state["Plan_GB"],
            "WiFi_Availability_Lag1": state["WiFi_Availability_Lag1"],
            "Is_Weekend": int(next_date.dayofweek >= 5),
            "Is_Holiday": 0,  # no forward holiday calendar beyond dataset scope
            "Occupation": state["Occupation"],
            "Network_Type": state["Network_Type"],
        }
        X_next = pd.DataFrame([feat])[ALL_FEATURES]
        pred = float(model.predict(X_next)[0])
        pred = max(pred, 0.0)
        preds.append({"Date": next_date, "Predicted_MB": pred})
        hist.append(pred)  # feed prediction forward for next step (recursive)
    return preds

# Build a lookup of each user's actual recent history (up to last 30 real days)
recent_history = {}
for uid, sub in df.groupby("User_ID"):
    recent_history[uid] = sub["Mobile_Data_MB"].tail(30).tolist()

forecast_records = []
for _, row in last_rows.iterrows():
    fc = recursive_forecast(row)
    for f in fc:
        forecast_records.append({"User_ID": row["User_ID"], **f})
forecast_df = pd.DataFrame(forecast_records)

# Day+1 prediction (used as the headline "Predicted_Next_Day_Usage_MB")
next_day_pred = (
    forecast_df[forecast_df.groupby("User_ID")["Date"].transform("min") == forecast_df["Date"]]
    .set_index("User_ID")["Predicted_MB"]
    .rename("Predicted_Next_Day_Usage_MB")
)

# Sum of the 7 forecast days = Predicted_7_Day_Usage_MB (never a naive x7 multiply)
seven_day_sum = forecast_df.groupby("User_ID")["Predicted_MB"].sum().rename("Predicted_7_Day_Usage_MB")

# ---------------------------------------------------------------------------
# 5. Month-end forecast = actual month-to-date + predicted remaining days
#    Remaining days in the current month are forecast recursively (reuse the
#    7-day recursive forecast if it covers them; extend if the month has
#    more remaining days than the 7-day horizon).
# ---------------------------------------------------------------------------
remaining_days = month_days_total - day_of_month

def month_end_forecast(row):
    if remaining_days <= 0:
        return 0.0
    if remaining_days <= FORECAST_HORIZON_DAYS:
        sub = forecast_df[forecast_df["User_ID"] == row["User_ID"]].sort_values("Date").head(remaining_days)
        return sub["Predicted_MB"].sum()
    # Need more days than our 7-day forecast covers: extend recursively.
    fc_full = recursive_forecast(row, horizon=remaining_days)
    return sum(f["Predicted_MB"] for f in fc_full)

remaining_forecast = last_rows.apply(month_end_forecast, axis=1)
remaining_forecast.index = last_rows["User_ID"].values
remaining_forecast = remaining_forecast.rename("Predicted_Remaining_Month_MB")

# ---------------------------------------------------------------------------
# 6. Assemble the prediction output table
# ---------------------------------------------------------------------------
pred_df = last_rows.set_index("User_ID")[[
    "Date", "Mobile_Data_MB", "Plan_GB", "Occupation", "Age", "Network_Type",
]].rename(columns={"Date": "Prediction_Date", "Mobile_Data_MB": "Current_Usage_MB"})

pred_df = pred_df.join(next_day_pred).join(seven_day_sum).join(mtd_actual).join(remaining_forecast)

pred_df["Current_Month_Usage_MB"] = pred_df["Current_Month_Usage_MB"].fillna(0)
pred_df["Predicted_Month_End_Usage_MB"] = pred_df["Current_Month_Usage_MB"] + pred_df["Predicted_Remaining_Month_MB"].fillna(0)

pred_df["Plan_MB"] = pred_df["Plan_GB"] * 1024
pred_df["Remaining_Plan_MB"] = np.clip(pred_df["Plan_MB"] - pred_df["Current_Month_Usage_MB"], 0, None)
pred_df["Expected_Shortage_MB"] = np.clip(pred_df["Predicted_Month_End_Usage_MB"] - pred_df["Plan_MB"], 0, None)

# ---------------------------------------------------------------------------
# 7. Risk classification (explainable, threshold-based on predicted % of plan)
# ---------------------------------------------------------------------------
pred_df["Predicted_Usage_Pct_Of_Plan"] = pred_df["Predicted_Month_End_Usage_MB"] / pred_df["Plan_MB"]

def classify_risk(pct):
    if pct <= RISK_LOW_MAX_PCT:
        return "Low"
    elif pct <= RISK_MEDIUM_MAX_PCT:
        return "Medium"
    else:
        return "High"

pred_df["Risk_Level"] = pred_df["Predicted_Usage_Pct_Of_Plan"].apply(classify_risk)

# ---------------------------------------------------------------------------
# 8. Expected exhaustion date — uses the FORECAST daily usage (not a naive
#    plan/average-usage division), walking forward day by day until the
#    remaining plan allowance is used up. Handles edge cases explicitly.
# ---------------------------------------------------------------------------
def exhaustion_date(user_id, remaining_plan_mb):
    if remaining_plan_mb <= 0:
        return "Already Exhausted"
    user_fc = forecast_df[forecast_df["User_ID"] == user_id].sort_values("Date")
    running = 0.0
    for _, r in user_fc.iterrows():
        running += r["Predicted_MB"]
        if running >= remaining_plan_mb:
            return r["Date"].strftime("%Y-%m-%d")
    # Not exhausted within the 7-day forecast horizon
    avg_daily = user_fc["Predicted_MB"].mean() if len(user_fc) > 0 else 0
    if avg_daily <= 0.01:
        return "Not Expected (Negligible Usage)"
    extra_days_needed = (remaining_plan_mb - running) / avg_daily
    if extra_days_needed > (month_days_total - day_of_month):
        return "Not Expected This Month"
    est_date = user_fc["Date"].max() + pd.Timedelta(days=int(np.ceil(extra_days_needed)))
    return est_date.strftime("%Y-%m-%d")

pred_df["Expected_Exhaustion_Date"] = [
    exhaustion_date(uid, rem) for uid, rem in zip(pred_df.index, pred_df["Remaining_Plan_MB"])
]

# ---------------------------------------------------------------------------
# 9. Recommendation engine (explainable rules on risk + shortage + trend)
# ---------------------------------------------------------------------------
def recommend(row):
    if row["Risk_Level"] == "High" and row["Expected_Shortage_MB"] > 0.5 * row["Plan_MB"]:
        return "Consider Higher Data Plan"
    if row["Risk_Level"] == "High":
        return "Consider Data Add-on"
    if row["Risk_Level"] == "Medium" and row["Predicted_Usage_Pct_Of_Plan"] > 0.90:
        return "Reduce Streaming Usage"
    if row["Risk_Level"] == "Medium":
        return "Monitor Usage"
    return "Keep Current Plan"

pred_df["Recommendation"] = pred_df.apply(recommend, axis=1)

pred_df = pred_df.reset_index().rename(columns={"index": "User_ID"})
final_cols = [
    "User_ID", "Prediction_Date", "Current_Usage_MB", "Predicted_Next_Day_Usage_MB",
    "Predicted_7_Day_Usage_MB", "Current_Month_Usage_MB", "Plan_GB", "Remaining_Plan_MB",
    "Predicted_Month_End_Usage_MB", "Expected_Shortage_MB", "Risk_Level",
    "Expected_Exhaustion_Date", "Recommendation",
]
pred_df["Prediction_Date"] = pd.to_datetime(pred_df["Prediction_Date"]).dt.strftime("%Y-%m-%d")
pred_out = pred_df[final_cols].round(2)

print("\nRisk level distribution:\n", pred_out["Risk_Level"].value_counts())
print("\nRecommendation distribution:\n", pred_out["Recommendation"].value_counts())

pred_out.to_csv(OUTPUT_DIR / "mobile_usage_predictions.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'mobile_usage_predictions.csv'} ({len(pred_out)} rows)")

# ---------------------------------------------------------------------------
# 10. What-if analysis: reduce streaming (YouTube + Netflix) usage by 20%
#     Since the model wasn't trained to take a modified app-level input
#     directly (its features are lag/rolling totals, not a live app split),
#     we implement a transparent, consistent scenario calculation:
#     re-derive Predicted_Month_End_Usage by removing 20% of the user's
#     recent streaming share from the forecast total.
# ---------------------------------------------------------------------------
streaming_share = (
    df[df["Year_Month"] == current_period]
    .groupby("User_ID")
    .apply(lambda s: (s["YouTube_MB"].sum() + s["Netflix_MB"].sum()) / s["Mobile_Data_MB"].sum() if s["Mobile_Data_MB"].sum() > 0 else 0)
    .rename("Streaming_Share_Of_Usage")
)

whatif = pred_out.set_index("User_ID").join(streaming_share)
whatif["Streaming_Share_Of_Usage"] = whatif["Streaming_Share_Of_Usage"].fillna(0)

whatif["Original_Predicted_Month_End_MB"] = whatif["Predicted_Month_End_Usage_MB"]
whatif["Data_Saved_MB"] = (
    whatif["Original_Predicted_Month_End_MB"] * whatif["Streaming_Share_Of_Usage"] * STREAMING_REDUCTION_PCT
)
whatif["New_Predicted_Month_End_MB"] = whatif["Original_Predicted_Month_End_MB"] - whatif["Data_Saved_MB"]

plan_mb = whatif["Plan_GB"] * 1024
whatif["Original_Shortage_MB"] = np.clip(whatif["Original_Predicted_Month_End_MB"] - plan_mb, 0, None)
whatif["New_Shortage_MB"] = np.clip(whatif["New_Predicted_Month_End_MB"] - plan_mb, 0, None)
whatif["Original_Risk"] = whatif["Risk_Level"]
whatif["New_Risk"] = (whatif["New_Predicted_Month_End_MB"] / plan_mb).apply(classify_risk)

whatif_out = whatif.reset_index()[[
    "User_ID", "Streaming_Share_Of_Usage", "Original_Predicted_Month_End_MB",
    "New_Predicted_Month_End_MB", "Data_Saved_MB", "Original_Shortage_MB",
    "New_Shortage_MB", "Original_Risk", "New_Risk",
]].round(2)

whatif_out.to_csv(OUTPUT_DIR / "whatif_streaming_reduction.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'whatif_streaming_reduction.csv'} ({len(whatif_out)} rows)")
print("\nRisk change summary (What-If: -20% streaming):")
print(pd.crosstab(whatif_out["Original_Risk"], whatif_out["New_Risk"]))
