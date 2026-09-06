"""
Step 5: Exploratory Data Analysis
Generates charts (PNG) from the cleaned dataset into project/eda_charts/
and prints the numeric findings each chart is based on.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_PATH = BASE_DIR / "data" / "cleaned" / "mobile_data_usage_cleaned.csv"
CHART_DIR = BASE_DIR / "eda_charts"
CHART_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
plt.rcParams["figure.figsize"] = (9, 5)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, dpi=110)
    plt.close(fig)

# 1. Daily mobile data usage trend (overall mean per day)
daily_trend = df.groupby("Date")["Mobile_Data_MB"].mean()
fig, ax = plt.subplots()
daily_trend.plot(ax=ax)
ax.set_title("Average Daily Mobile Data Usage (All Users)")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "01_daily_usage_trend.png")

# 2. Weekly usage pattern (by day of week)
df["DOW"] = df["Date"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly_pattern = df.groupby("DOW")["Mobile_Data_MB"].mean().reindex(dow_order)
fig, ax = plt.subplots()
weekly_pattern.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average Mobile Data Usage by Day of Week")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "02_weekly_pattern.png")

# 3. Weekend vs weekday usage
weekend_cmp = df.groupby("Is_Weekend")["Mobile_Data_MB"].mean().rename({0: "Weekday", 1: "Weekend"})
fig, ax = plt.subplots()
weekend_cmp.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("Weekend vs Weekday Average Usage")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "03_weekend_vs_weekday.png")

# 4. Usage by occupation
occ_usage = df.groupby("Occupation")["Mobile_Data_MB"].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
occ_usage.plot(kind="bar", ax=ax, color="seagreen")
ax.set_title("Average Mobile Data Usage by Occupation")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "04_usage_by_occupation.png")

# 5. Usage by network type
net_usage = df.groupby("Network_Type")["Mobile_Data_MB"].mean()
fig, ax = plt.subplots()
net_usage.plot(kind="bar", ax=ax, color="indianred")
ax.set_title("Average Mobile Data Usage by Network Type")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "05_usage_by_network_type.png")

# 6. Usage by age group
bins = [0, 20, 30, 40, 50, 60, 100]
labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
age_usage = df.groupby("Age_Group", observed=True)["Mobile_Data_MB"].mean()
fig, ax = plt.subplots()
age_usage.plot(kind="bar", ax=ax, color="mediumpurple")
ax.set_title("Average Mobile Data Usage by Age Group")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "06_usage_by_age_group.png")

# 7. App/category consumption (overall average share)
app_cols = ["YouTube_MB", "Netflix_MB", "Instagram_MB", "WhatsApp_MB", "Gaming_MB", "Browsing_MB", "Work_Study_MB"]
app_avg = df[app_cols].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
app_avg.plot(kind="bar", ax=ax, color="darkorange")
ax.set_title("Average Daily Usage by App/Category")
ax.set_ylabel("MB")
save(fig, "07_app_category_consumption.png")

# 8. Screen time vs mobile data (scatter, sampled for readability)
sample = df.sample(3000, random_state=42)
fig, ax = plt.subplots()
ax.scatter(sample["Screen_Time_Hours"], sample["Mobile_Data_MB"], alpha=0.25, s=10)
ax.set_xlabel("Screen_Time_Hours")
ax.set_ylabel("Mobile_Data_MB")
ax.set_title("Screen Time vs Mobile Data Usage")
save(fig, "08_screen_time_vs_usage.png")
corr_screen = df[["Screen_Time_Hours", "Mobile_Data_MB"]].corr().iloc[0, 1]

# 9. Wi-Fi availability vs mobile usage
wifi_cmp = df.groupby("WiFi_Availability")["Mobile_Data_MB"].mean().rename({0: "No WiFi", 1: "WiFi Available"})
fig, ax = plt.subplots()
wifi_cmp.plot(kind="bar", ax=ax, color=["#C44E52", "#55A868"])
ax.set_title("Wi-Fi Availability vs Mobile Data Usage")
ax.set_ylabel("Mobile_Data_MB")
save(fig, "09_wifi_vs_usage.png")

# 10. Distribution of daily usage
fig, ax = plt.subplots()
ax.hist(df["Mobile_Data_MB"], bins=50, color="cornflowerblue", edgecolor="white")
ax.set_title("Distribution of Daily Mobile Data Usage")
ax.set_xlabel("Mobile_Data_MB")
save(fig, "10_usage_distribution.png")

# 11 & 12. Highest / lowest usage customers (by average daily usage)
user_avg = df.groupby("User_ID")["Mobile_Data_MB"].mean().sort_values(ascending=False)
top10 = user_avg.head(10)
bottom10 = user_avg.tail(10)

fig, ax = plt.subplots()
top10.plot(kind="barh", ax=ax, color="crimson")
ax.set_title("Top 10 Highest-Usage Customers (Avg Daily MB)")
ax.invert_yaxis()
save(fig, "11_top10_highest_usage.png")

fig, ax = plt.subplots()
bottom10.plot(kind="barh", ax=ax, color="teal")
ax.set_title("Bottom 10 Lowest-Usage Customers (Avg Daily MB)")
ax.invert_yaxis()
save(fig, "12_bottom10_lowest_usage.png")

# ---------------------------------------------------------------------------
# Print the calculated findings (used verbatim in the notebook narrative)
# ---------------------------------------------------------------------------
print("=== EDA FINDINGS (calculated from actual data) ===")
print(f"Overall average daily mobile usage: {df['Mobile_Data_MB'].mean():.1f} MB")
print(f"Weekend avg: {weekend_cmp['Weekend']:.1f} MB | Weekday avg: {weekend_cmp['Weekday']:.1f} MB "
      f"({(weekend_cmp['Weekend']/weekend_cmp['Weekday']-1)*100:.1f}% higher on weekends)")
print("\nUsage by occupation:\n", occ_usage.round(1))
print("\nUsage by network type:\n", net_usage.round(1))
print("\nUsage by age group:\n", age_usage.round(1))
print("\nApp/category average consumption (MB/day):\n", app_avg.round(1))
print(f"\nCorrelation(Screen_Time_Hours, Mobile_Data_MB) = {corr_screen:.3f}")
print(f"\nWiFi available avg: {wifi_cmp['WiFi Available']:.1f} MB | No WiFi avg: {wifi_cmp['No WiFi']:.1f} MB "
      f"({(1 - wifi_cmp['WiFi Available']/wifi_cmp['No WiFi'])*100:.1f}% lower usage when WiFi is available)")
print("\nTop 10 highest-usage customers (avg MB/day):\n", top10.round(1))
print("\nBottom 10 lowest-usage customers (avg MB/day):\n", bottom10.round(1))
