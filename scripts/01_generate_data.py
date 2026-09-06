"""
Smart Mobile Data Usage Prediction & Plan Recommendation System
Step 1: Synthetic Dataset Generation

Generates a realistic, internally-consistent telecom daily-usage dataset for
500 users over 180 consecutive days (~90,000 rows), then injects controlled,
realistic data-quality issues (missing values, duplicates, minor outliers)
for the cleaning stage to fix later.

Output:
    data/raw/mobile_data_usage.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_USERS = 500
N_DAYS = 180
START_DATE = pd.Timestamp("2025-01-01")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OCCUPATIONS = ["Student", "Working Professional", "Self-Employed", "Homemaker", "Retired"]
OCC_WEIGHTS = [0.30, 0.35, 0.15, 0.12, 0.08]

NETWORK_TYPES = ["4G", "5G"]

PLAN_OPTIONS = [  # (Plan_GB, Plan_Price)
    (1.5, 199), (2.0, 249), (3.0, 349), (5.0, 499), (10.0, 799),
]

# ---------------------------------------------------------------------------
# 1. Build user profiles (one row per user) — this is what creates
#    realistic, *persistent* behavioural differences between users instead
#    of pure day-to-day randomness.
# ---------------------------------------------------------------------------
user_ids = [f"U{str(i).zfill(4)}" for i in range(1, N_USERS + 1)]

profiles = pd.DataFrame({"User_ID": user_ids})
profiles["Age"] = rng.integers(16, 66, size=N_USERS)
profiles["Occupation"] = rng.choice(OCCUPATIONS, size=N_USERS, p=OCC_WEIGHTS)

# Students skew younger; retired skew older -> nudge ages to be consistent
profiles.loc[profiles["Occupation"] == "Student", "Age"] = rng.integers(16, 26, size=(profiles["Occupation"] == "Student").sum())
profiles.loc[profiles["Occupation"] == "Retired", "Age"] = rng.integers(58, 76, size=(profiles["Occupation"] == "Retired").sum())
profiles.loc[profiles["Occupation"] == "Working Professional", "Age"] = rng.integers(23, 55, size=(profiles["Occupation"] == "Working Professional").sum())

# Network type: 5G more common among younger / professional users
p_5g = np.where(profiles["Occupation"].isin(["Student", "Working Professional"]), 0.55, 0.30)
profiles["Network_Type"] = [rng.choice(NETWORK_TYPES, p=[1 - p, p]) for p in p_5g]

# Persistent "usage personality" per user: low / medium / high baseline
# Students & professionals skew to medium/high usage; retired/homemaker skew low
usage_tier_probs = {
    "Student": [0.15, 0.45, 0.40],
    "Working Professional": [0.20, 0.50, 0.30],
    "Self-Employed": [0.25, 0.50, 0.25],
    "Homemaker": [0.40, 0.45, 0.15],
    "Retired": [0.55, 0.35, 0.10],
}
tiers = ["Low", "Medium", "High"]
profiles["Usage_Tier"] = [
    rng.choice(tiers, p=usage_tier_probs[occ]) for occ in profiles["Occupation"]
]

tier_baseline_mb = {"Low": 350, "Medium": 750, "High": 1400}
profiles["Baseline_Mobile_MB"] = profiles["Usage_Tier"].map(tier_baseline_mb) * rng.uniform(0.8, 1.2, size=N_USERS)

# Streaming affinity (0-1): how entertainment-heavy the user is (drives Netflix/YouTube weight)
profiles["Streaming_Affinity"] = np.clip(rng.normal(0.5, 0.2, size=N_USERS), 0.05, 0.95)
# Gaming affinity
profiles["Gaming_Affinity"] = np.clip(rng.normal(0.3, 0.2, size=N_USERS), 0.02, 0.9)
# Work/study intensity (professionals & students higher)
profiles["Work_Study_Affinity"] = np.where(
    profiles["Occupation"].isin(["Student", "Working Professional"]),
    np.clip(rng.normal(0.6, 0.15, size=N_USERS), 0.1, 0.95),
    np.clip(rng.normal(0.25, 0.15, size=N_USERS), 0.02, 0.7),
)
# Baseline WiFi availability probability (home/office wifi quality varies by user)
profiles["WiFi_Prob"] = np.clip(rng.normal(0.55, 0.18, size=N_USERS), 0.05, 0.95)

# Plan assignment: higher usage tier -> tends to a bigger plan, with some noise
def assign_plan(tier, rngen):
    if tier == "Low":
        weights = [0.45, 0.30, 0.15, 0.08, 0.02]
    elif tier == "Medium":
        weights = [0.10, 0.25, 0.35, 0.22, 0.08]
    else:
        weights = [0.03, 0.07, 0.20, 0.35, 0.35]
    idx = rngen.choice(len(PLAN_OPTIONS), p=weights)
    return PLAN_OPTIONS[idx]

plan_choices = [assign_plan(t, rng) for t in profiles["Usage_Tier"]]
profiles["Plan_GB"] = [p[0] for p in plan_choices]
profiles["Plan_Price"] = [p[1] for p in plan_choices]

# ---------------------------------------------------------------------------
# 2. Build the daily panel (User x Date) and simulate day-level behaviour
# ---------------------------------------------------------------------------
dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

panel = pd.MultiIndex.from_product([user_ids, dates], names=["User_ID", "Date"]).to_frame(index=False)
panel = panel.merge(profiles, on="User_ID", how="left")

n = len(panel)
assert n == N_USERS * N_DAYS

panel["Is_Weekend"] = panel["Date"].dt.dayofweek.isin([5, 6]).astype(int)

# A small set of shared holidays across the period (realistic, sparse)
holiday_dates = pd.to_datetime([
    "2025-01-01", "2025-01-14", "2025-01-26", "2025-02-14", "2025-03-14",
    "2025-03-31", "2025-04-14", "2025-05-01", "2025-06-07",
])
panel["Is_Holiday"] = panel["Date"].isin(holiday_dates).astype(int)

# Day index per user (for mild trend / novelty effects, e.g. new-plan excitement fading)
panel["Day_Index"] = panel.groupby("User_ID").cumcount()

# ---- WiFi availability (daily, correlated with each user's WiFi_Prob) ----
panel["WiFi_Availability"] = (rng.uniform(size=n) < panel["WiFi_Prob"]).astype(int)

# ---- Screen time (hours): driven by tier baseline, weekend boost, some noise ----
weekend_boost = np.where(panel["Is_Weekend"] == 1, rng.uniform(0.3, 1.0, size=n), 0.0)
holiday_boost = np.where(panel["Is_Holiday"] == 1, rng.uniform(0.2, 0.8, size=n), 0.0)
tier_screen_base = panel["Usage_Tier"].map({"Low": 2.2, "Medium": 3.6, "High": 5.2}).to_numpy()
screen_noise = rng.normal(0, 0.6, size=n)
panel["Screen_Time_Hours"] = np.clip(tier_screen_base + weekend_boost + holiday_boost + screen_noise, 0.3, 12.0)

# ---------------------------------------------------------------------------
# 3. App-level components first (MB), THEN derive Mobile_Data_MB so totals
#    are always internally consistent (component sum <= Mobile_Data_MB).
# ---------------------------------------------------------------------------
screen_factor = panel["Screen_Time_Hours"] / panel["Screen_Time_Hours"].groupby(panel["User_ID"]).transform("mean").clip(lower=0.5)
weekend_mult = np.where(panel["Is_Weekend"] == 1, rng.uniform(1.05, 1.35, size=n), 1.0)

# Wi-Fi offload: when Wi-Fi is available, a large share of streaming/browsing
# is offloaded to Wi-Fi instead of mobile data.
wifi_offload = np.where(panel["WiFi_Availability"] == 1, rng.uniform(0.55, 0.85, size=n), rng.uniform(0.0, 0.15, size=n))

def app_component(base_scale, affinity_col, weekend_sensitivity=1.0, evening_effect=1.0):
    affinity = panel[affinity_col].to_numpy() if affinity_col else 1.0
    raw = (
        base_scale
        * affinity
        * screen_factor.to_numpy()
        * (1 + (weekend_mult - 1) * weekend_sensitivity)
        * evening_effect
        * rng.lognormal(mean=0, sigma=0.35, size=n)
    )
    # Wi-Fi offload reduces the MOBILE portion of streaming/entertainment apps
    mobile_share = 1 - wifi_offload
    return np.clip(raw * mobile_share, 0, None)

# Evening usage effect: apply a modest uplift baked into an "evening intensity"
# factor per row (simulating that most sessions cluster in the evening).
evening_intensity = rng.uniform(0.9, 1.25, size=n)

youtube_mb = app_component(180, "Streaming_Affinity", weekend_sensitivity=1.2, evening_effect=evening_intensity)
netflix_mb = app_component(220, "Streaming_Affinity", weekend_sensitivity=1.4, evening_effect=evening_intensity) * \
             np.where(panel["Streaming_Affinity"] > 0.35, 1.0, 0.15)  # low-affinity users barely use Netflix
instagram_mb = app_component(90, "Streaming_Affinity", weekend_sensitivity=1.1, evening_effect=evening_intensity) * 0.8 \
               + app_component(40, None, weekend_sensitivity=1.0, evening_effect=1.0)
whatsapp_mb = app_component(15, None, weekend_sensitivity=0.9, evening_effect=1.0) + rng.uniform(2, 10, size=n)
gaming_mb = app_component(120, "Gaming_Affinity", weekend_sensitivity=1.3, evening_effect=evening_intensity)
browsing_mb = app_component(60, None, weekend_sensitivity=1.0, evening_effect=1.0) + rng.uniform(5, 20, size=n)
work_study_mb = app_component(140, "Work_Study_Affinity", weekend_sensitivity=0.4, evening_effect=1.0)
# Work/study usage drops sharply on weekends & holidays (office/school apps, docs, video calls)
work_study_mb = work_study_mb * np.where((panel["Is_Weekend"] == 1) | (panel["Is_Holiday"] == 1), 0.35, 1.0)

# Miscellaneous / other background mobile usage (OS updates, notifications, misc apps)
misc_mb = np.clip(rng.normal(60, 20, size=n) * (1 - wifi_offload * 0.5), 5, None)

app_sum = youtube_mb + netflix_mb + instagram_mb + whatsapp_mb + gaming_mb + browsing_mb + work_study_mb

# Mobile_Data_MB = tracked app components + misc, scaled by each user's
# persistent baseline usage level so totals vary sensibly user-to-user.
baseline_scale = (panel["Baseline_Mobile_MB"] / 750.0).to_numpy()  # normalised around "Medium" tier
mobile_data_mb = (app_sum * 0.55 + misc_mb) * np.clip(baseline_scale, 0.4, 2.2)
mobile_data_mb = np.clip(mobile_data_mb, 20, None)

# Re-derive the app columns proportionally so that sum(app columns) <= Mobile_Data_MB
# (misc is not stored as its own column; it's absorbed into Mobile_Data_MB).
scale_to_fit = np.minimum(1.0, (mobile_data_mb * 0.92) / np.clip(app_sum, 1e-6, None))
youtube_mb = youtube_mb * scale_to_fit
netflix_mb = netflix_mb * scale_to_fit
instagram_mb = instagram_mb * scale_to_fit
whatsapp_mb = whatsapp_mb * scale_to_fit
gaming_mb = gaming_mb * scale_to_fit
browsing_mb = browsing_mb * scale_to_fit
work_study_mb = work_study_mb * scale_to_fit

panel["YouTube_MB"] = youtube_mb.round(2)
panel["Netflix_MB"] = netflix_mb.round(2)
panel["Instagram_MB"] = instagram_mb.round(2)
panel["WhatsApp_MB"] = whatsapp_mb.round(2)
panel["Gaming_MB"] = gaming_mb.round(2)
panel["Browsing_MB"] = browsing_mb.round(2)
panel["Work_Study_MB"] = work_study_mb.round(2)
panel["Mobile_Data_MB"] = mobile_data_mb.round(2)

# Final consistency guard: ensure app sum never exceeds Mobile_Data_MB
app_cols = ["YouTube_MB", "Netflix_MB", "Instagram_MB", "WhatsApp_MB", "Gaming_MB", "Browsing_MB", "Work_Study_MB"]
check_sum = panel[app_cols].sum(axis=1)
violation = check_sum > panel["Mobile_Data_MB"]
assert violation.sum() == 0, f"{violation.sum()} rows violate app-sum <= Mobile_Data_MB"

# ---------------------------------------------------------------------------
# 3b. Re-assign plans based on each user's ACTUAL simulated average usage.
#     Plan choice at the profile stage only used the coarse usage tier, which
#     produced an unrealistic mismatch between plans and actual simulated
#     consumption. Real telecom customers mostly buy a plan that roughly
#     covers their needs, with a meaningful minority under-provisioned
#     (they underestimate usage, or usage grew after they picked the plan).
#     This produces a believable Low/Medium/High risk mix downstream.
# ---------------------------------------------------------------------------
avg_daily_mobile = pd.Series(mobile_data_mb).groupby(panel["User_ID"].values).mean()
expected_monthly_mb = avg_daily_mobile * 30

buffer_factor = np.where(
    rng.uniform(size=len(avg_daily_mobile)) < 0.18,
    rng.uniform(0.55, 0.95, size=len(avg_daily_mobile)),   # under-provisioned (~18% of users)
    rng.uniform(1.05, 1.45, size=len(avg_daily_mobile)),   # comfortable buffer
)
target_plan_mb = expected_monthly_mb.to_numpy() * buffer_factor

plan_gb_options = np.array([p[0] for p in PLAN_OPTIONS])
plan_price_options = np.array([p[1] for p in PLAN_OPTIONS])
plan_mb_options = plan_gb_options * 1024

def pick_plan(target_mb):
    idx = np.searchsorted(plan_mb_options, target_mb)
    idx = min(idx, len(plan_mb_options) - 1)
    return plan_gb_options[idx], plan_price_options[idx]

picked = [pick_plan(t) for t in target_plan_mb]
new_plan_gb = pd.Series([p[0] for p in picked], index=avg_daily_mobile.index)
new_plan_price = pd.Series([p[1] for p in picked], index=avg_daily_mobile.index)

panel["Plan_GB"] = panel["User_ID"].map(new_plan_gb)
panel["Plan_Price"] = panel["User_ID"].map(new_plan_price)

# ---- WiFi_Data_MB: usage that happened over Wi-Fi (separate from mobile) ----
wifi_data_mb = np.where(
    panel["WiFi_Availability"] == 1,
    (app_sum / scale_to_fit.clip(min=1e-6)) * wifi_offload + rng.uniform(50, 400, size=n),
    rng.uniform(0, 30, size=n),
)
panel["WiFi_Data_MB"] = np.clip(wifi_data_mb, 0, None).round(2)

# ---------------------------------------------------------------------------
# 4. Finalize column order & types
# ---------------------------------------------------------------------------
panel["Date"] = panel["Date"].dt.strftime("%Y-%m-%d")

final_cols = [
    "User_ID", "Date", "Age", "Occupation", "Plan_GB", "Plan_Price",
    "Network_Type", "WiFi_Availability", "Mobile_Data_MB", "WiFi_Data_MB",
    "YouTube_MB", "Instagram_MB", "WhatsApp_MB", "Netflix_MB", "Gaming_MB",
    "Browsing_MB", "Work_Study_MB", "Screen_Time_Hours", "Is_Weekend", "Is_Holiday",
]
df = panel[final_cols].copy()
df["Screen_Time_Hours"] = df["Screen_Time_Hours"].round(2)

print("Clean (pre-noise) dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------------------
# 5. Inject controlled, realistic data-quality issues (for the cleaning stage)
# ---------------------------------------------------------------------------
df_raw = df.copy()

# 5a. Missing values (~0.6% of numeric cells, scattered across a few columns)
missing_cols = ["Mobile_Data_MB", "Screen_Time_Hours", "WiFi_Data_MB", "Age"]
for col in missing_cols:
    frac = 0.006 if col != "Age" else 0.003
    n_missing = int(len(df_raw) * frac)
    idx = rng.choice(df_raw.index, size=n_missing, replace=False)
    df_raw.loc[idx, col] = np.nan

# 5b. Duplicate records (~0.3% of rows duplicated, simulating double-logging)
n_dupe = int(len(df_raw) * 0.003)
dupe_rows = df_raw.sample(n=n_dupe, random_state=SEED)
df_raw = pd.concat([df_raw, dupe_rows], ignore_index=True)

# 5c. Minor outliers (~0.2% of Mobile_Data_MB rows get an unrealistic spike)
outlier_idx = rng.choice(df_raw.index, size=int(len(df_raw) * 0.002), replace=False)
df_raw.loc[outlier_idx, "Mobile_Data_MB"] = df_raw.loc[outlier_idx, "Mobile_Data_MB"] * rng.uniform(4, 8, size=len(outlier_idx))

# Shuffle row order slightly (real logs aren't perfectly sorted) but keep it reproducible
df_raw = df_raw.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("\nRaw (with quality issues) dataset shape:", df_raw.shape)
print("Missing values per column:\n", df_raw.isna().sum()[df_raw.isna().sum() > 0])
print("Duplicate (User_ID, Date) rows:", df_raw.duplicated(subset=["User_ID", "Date"]).sum())

out_path = RAW_DIR / "mobile_data_usage.csv"
df_raw.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(df_raw)} rows)")
