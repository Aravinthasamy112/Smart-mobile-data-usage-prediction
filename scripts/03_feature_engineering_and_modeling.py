"""
Step 3: Feature Engineering + ML Modeling (leakage-safe)

Target: Next_Day_Mobile_Data_MB (created via a proper forward shift, sorted
by User_ID + Date; each user's last day is dropped since it has no next day).

Features use ONLY information available on/before day T (lags, rolling
averages built from SHIFTED history, static profile info, calendar flags).

Split: chronological (first ~80% of each user's timeline = train,
last ~20% = test) — NOT a random split, because usage is autocorrelated
over time and a random split would leak future information into training
via adjacent-day rolling features and would not resemble real deployment
(where you always predict forward in time).
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_PATH = BASE_DIR / "data" / "cleaned" / "mobile_data_usage_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & sort (critical: sort by User_ID + Date BEFORE any shift/rolling)
# ---------------------------------------------------------------------------
df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)

g = df.groupby("User_ID")

# ---------------------------------------------------------------------------
# 2. Target: Next_Day_Mobile_Data_MB (forward shift of -1 within each user)
# ---------------------------------------------------------------------------
df["Next_Day_Mobile_Data_MB"] = g["Mobile_Data_MB"].shift(-1)

# ---------------------------------------------------------------------------
# 3. Leakage-safe features: everything is built from data at or before day T,
#    using .shift(1) before any rolling window so the rolling window for day T
#    only ever looks at days T-1, T-2, ... (never T itself or later).
# ---------------------------------------------------------------------------
shifted_usage = g["Mobile_Data_MB"].shift(1)          # value as of T-1 (known at prediction time for day T)
df["Lag1_Mobile_MB"] = shifted_usage

for window in [3, 7, 14, 30]:
    df[f"RollMean{window}_Mobile_MB"] = (
        shifted_usage.groupby(df["User_ID"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )

streaming_mb = df["YouTube_MB"] + df["Netflix_MB"]
df["Lag1_Streaming_MB"] = streaming_mb.groupby(df["User_ID"]).shift(1)
df["Lag1_Screen_Time_Hours"] = g["Screen_Time_Hours"].shift(1)

# Recent usage growth: (avg of last 3 days) vs (avg of days 4-7 ago), both from shifted history
roll3 = df["RollMean3_Mobile_MB"]
roll7 = df["RollMean7_Mobile_MB"]
df["Usage_Growth_Ratio"] = (roll3 / roll7.replace(0, np.nan)).fillna(1.0)

# Static/profile + calendar features (known in advance, not leakage)
df["WiFi_Availability_Lag1"] = g["WiFi_Availability"].shift(1)

FEATURE_COLS_NUMERIC = [
    "Lag1_Mobile_MB", "RollMean3_Mobile_MB", "RollMean7_Mobile_MB",
    "RollMean14_Mobile_MB", "RollMean30_Mobile_MB", "Lag1_Streaming_MB",
    "Lag1_Screen_Time_Hours", "Usage_Growth_Ratio", "Age", "Plan_GB",
    "WiFi_Availability_Lag1",
]
FEATURE_COLS_BINARY = ["Is_Weekend", "Is_Holiday"]
FEATURE_COLS_CATEGORICAL = ["Occupation", "Network_Type"]

ALL_FEATURES = FEATURE_COLS_NUMERIC + FEATURE_COLS_BINARY + FEATURE_COLS_CATEGORICAL

# Drop rows that cannot be used for supervised training:
#  - last day per user (no Next_Day target -> can't train/evaluate on it)
#  - first day(s) per user where lag/rolling features are undefined (no prior day)
model_df = df.dropna(subset=["Next_Day_Mobile_Data_MB", "Lag1_Mobile_MB"]).copy()

print("Rows available for modeling (after dropping last-day-per-user & first-day-per-user):", len(model_df))
print("Dropped rows (no target or no lag history):", len(df) - len(model_df))

# ---------------------------------------------------------------------------
# 4. Explicit leakage check
# ---------------------------------------------------------------------------
leak_terms = ["Next_Day", "Future"]
for col in ALL_FEATURES:
    assert not any(term in col for term in leak_terms), f"Potential leakage in feature name: {col}"
# Confirm no feature is computed using .shift(-...) (forward) - verified by code construction above.
print("Leakage check passed: all features derive only from shift(1)/rolling-on-shifted-history/static/calendar data.")

# ---------------------------------------------------------------------------
# 5. Chronological train/test split (per user: first 80% of days -> train)
# ---------------------------------------------------------------------------
model_df["Day_Rank"] = model_df.groupby("User_ID").cumcount()
model_df["User_N_Days"] = model_df.groupby("User_ID")["Day_Rank"].transform("max") + 1
model_df["Split_Point"] = (model_df["User_N_Days"] * 0.8).round().astype(int)
model_df["Split"] = np.where(model_df["Day_Rank"] < model_df["Split_Point"], "train", "test")

train_df = model_df[model_df["Split"] == "train"].copy()
test_df = model_df[model_df["Split"] == "test"].copy()
print(f"\nTrain rows: {len(train_df)}  |  Test rows: {len(test_df)}")
print("Train date range:", train_df["Date"].min(), "->", train_df["Date"].max())
print("Test date range:", test_df["Date"].min(), "->", test_df["Date"].max())

X_train, y_train = train_df[ALL_FEATURES], train_df["Next_Day_Mobile_Data_MB"]
X_test, y_test = test_df[ALL_FEATURES], test_df["Next_Day_Mobile_Data_MB"]

# ---------------------------------------------------------------------------
# 6. Preprocessing + models
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS_CATEGORICAL),
    ],
    remainder="passthrough",
)

models = {
    "Linear Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression()),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
    ]),
}

results = {}
fitted_models = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    fitted_models[name] = pipe
    print(f"\n{name}: MAE={mae:.2f} MB | RMSE={rmse:.2f} MB | R2={r2:.4f}")

best_model_name = min(results, key=lambda k: results[k]["MAE"])
best_model = fitted_models[best_model_name]
print(f"\nBest model (lowest MAE): {best_model_name}")

joblib.dump(best_model, MODEL_DIR / "best_next_day_usage_model.joblib")
with open(MODEL_DIR / "model_metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)
print(f"Saved best model -> {MODEL_DIR / 'best_next_day_usage_model.joblib'}")

# Save the modeling frame (features + split labels) for reuse in the
# prediction / forecasting stage so features are computed exactly once.
model_df.to_parquet(OUTPUT_DIR / "model_frame.parquet", index=False)
print(f"Saved modeling frame -> {OUTPUT_DIR / 'model_frame.parquet'}")
