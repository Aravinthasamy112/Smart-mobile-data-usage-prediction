"""
Step 2: Data Cleaning
Reads data/raw/mobile_data_usage.csv, performs inspection + cleaning,
writes data/cleaned/mobile_data_usage_cleaned.csv. Raw file is left untouched.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "mobile_data_usage.csv"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# 1. Import ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print("Loaded shape:", df.shape)

# 2. Shape / dtypes inspection ---------------------------------------------
print("\nDtypes:\n", df.dtypes)

# 3. Missing-value analysis -------------------------------------------------
missing = df.isna().sum()
print("\nMissing values:\n", missing[missing > 0])

# 4. Duplicate detection ------------------------------------------------
dupe_full = df.duplicated().sum()
dupe_key = df.duplicated(subset=["User_ID", "Date"]).sum()
print(f"\nFully duplicated rows: {dupe_full}")
print(f"Duplicated (User_ID, Date) rows: {dupe_key}")

# 5. Date conversion ------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])

# 6. Deduplicate ------------------------------------------------------------
# Keep the first occurrence for each (User_ID, Date); drop exact/partial repeats.
before = len(df)
df = df.sort_values(["User_ID", "Date"]).drop_duplicates(subset=["User_ID", "Date"], keep="first")
print(f"\nDropped {before - len(df)} duplicate (User_ID, Date) rows")

# 7. Missing-value handling --------------------------------------------------
# Age: static per user -> fill from the user's own mode (most common recorded age)
df["Age"] = df.groupby("User_ID")["Age"].transform(lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s.median()))

# Numeric usage columns: fill with the user's own rolling/user-level median
# (per-user median preserves each user's personal baseline instead of blending
# users together with a global average).
for col in ["Mobile_Data_MB", "Screen_Time_Hours", "WiFi_Data_MB"]:
    df[col] = df.groupby("User_ID")[col].transform(lambda s: s.fillna(s.median()))
    # fallback: any still-missing (e.g., user had all-NaN) -> global median
    df[col] = df[col].fillna(df[col].median())

remaining_missing = df.isna().sum().sum()
print(f"\nRemaining missing values after imputation: {remaining_missing}")

# 8. Range validation ---------------------------------------------------
range_checks = {
    "Age": (10, 90),
    "Screen_Time_Hours": (0, 24),
    "Mobile_Data_MB": (0, None),
    "WiFi_Data_MB": (0, None),
    "Plan_GB": (0, None),
    "Plan_Price": (0, None),
}
for col, (lo, hi) in range_checks.items():
    bad = df[(df[col] < lo) | (hi is not None and df[col] > hi)] if hi is not None else df[df[col] < lo]
    if len(bad) > 0:
        print(f"Range violation in {col}: {len(bad)} rows -> clipping")
        df[col] = df[col].clip(lower=lo, upper=hi)

# 9. Outlier detection & treatment (IQR method, applied globally on the
#    usage columns most prone to injected spikes) -----------------------
def iqr_bounds(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_cols = ["Mobile_Data_MB", "WiFi_Data_MB"]
outlier_report = {}
for col in outlier_cols:
    lo, hi = iqr_bounds(df[col])
    n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
    outlier_report[col] = (lo, hi, n_outliers)
    # Cap (winsorize) rather than drop, to avoid losing otherwise-valid rows
    df[col] = df[col].clip(lower=max(lo, 0), upper=hi)

print("\nOutlier bounds & counts (IQR k=3):")
for col, (lo, hi, cnt) in outlier_report.items():
    print(f"  {col}: bounds=({lo:.1f}, {hi:.1f}), flagged={cnt}")

# 10. Data consistency checks ------------------------------------------
app_cols = ["YouTube_MB", "Instagram_MB", "WhatsApp_MB", "Netflix_MB", "Gaming_MB", "Browsing_MB", "Work_Study_MB"]
app_sum = df[app_cols].sum(axis=1)
inconsistent = app_sum > df["Mobile_Data_MB"]
print(f"\nRows where app-usage sum exceeds Mobile_Data_MB (pre-fix): {inconsistent.sum()}")
if inconsistent.sum() > 0:
    # If capping Mobile_Data_MB created an inconsistency, rescale that row's
    # app components proportionally so the sum fits within the (now-capped) total.
    scale = np.where(inconsistent, (df["Mobile_Data_MB"] * 0.95) / app_sum.replace(0, np.nan), 1.0)
    scale = pd.Series(scale, index=df.index).fillna(1.0)
    for col in app_cols:
        df[col] = df[col] * scale
    app_sum_after = df[app_cols].sum(axis=1)
    print("Rows inconsistent after fix:", (app_sum_after > df["Mobile_Data_MB"]).sum())

# Negative-value guard (should be none, but validate defensively)
numeric_cols = ["Mobile_Data_MB", "WiFi_Data_MB"] + app_cols + ["Screen_Time_Hours"]
neg_counts = (df[numeric_cols] < 0).sum().sum()
print(f"Negative values remaining: {neg_counts}")
df[numeric_cols] = df[numeric_cols].clip(lower=0)

# Categorical validity
assert df["Is_Weekend"].isin([0, 1]).all()
assert df["Is_Holiday"].isin([0, 1]).all()
assert df["WiFi_Availability"].isin([0, 1]).all()

# 11. Final validation ----------------------------------------------------
assert df.duplicated(subset=["User_ID", "Date"]).sum() == 0, "Duplicate (User_ID, Date) remain!"
assert df.isna().sum().sum() == 0, "Missing values remain!"
assert (df[app_cols].sum(axis=1) <= df["Mobile_Data_MB"] + 1e-6).all(), "App-sum still exceeds Mobile_Data_MB!"

# Verify chronological order per user
is_sorted = df.groupby("User_ID")["Date"].apply(lambda s: s.is_monotonic_increasing).all()
assert is_sorted, "Dates are not chronological within a user!"

df = df.sort_values(["User_ID", "Date"]).reset_index(drop=True)

print("\nFinal cleaned shape:", df.shape)
out_path = CLEAN_DIR / "mobile_data_usage_cleaned.csv"
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

