"""
eda_trends.py — Exploratory Data Analysis & KPI Trend Reporting
==================================================================
Loads the retail transactions dataset and produces:
  1. A KPI summary (total revenue, AOV, transaction count, YoY growth)
  2. Monthly revenue trend chart (with seasonality visible)
  3. Revenue by category chart
  4. Revenue by region chart
  5. A consolidated CSV KPI report for downstream reporting/dashboarding

Usage:
    python eda_trends.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retail_transactions.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)
df["year"] = df["date"].dt.year

# ── 1. KPI Summary ────────────────────────────────────────────────
total_revenue = df["revenue"].sum()
total_txns = len(df)
aov = total_revenue / total_txns
unique_customers = df["customer_id"].nunique()

yearly_rev = df.groupby("year")["revenue"].sum()
yoy_growth = None
if len(yearly_rev) >= 2:
    yoy_growth = (yearly_rev.iloc[-1] / yearly_rev.iloc[0] - 1) * 100

kpi_summary = pd.DataFrame({
    "metric": [
        "Total Revenue", "Total Transactions", "Unique Customers",
        "Average Order Value", "YoY Revenue Growth (%)",
    ],
    "value": [
        round(total_revenue, 2), total_txns, unique_customers,
        round(aov, 2), round(yoy_growth, 2) if yoy_growth is not None else "N/A",
    ],
})
kpi_summary.to_csv(OUT_DIR / "kpi_summary.csv", index=False)
print("── KPI Summary ──")
print(kpi_summary.to_string(index=False))

# ── 2. Monthly Revenue Trend ──────────────────────────────────────
monthly = df.groupby("month")["revenue"].sum().reset_index()

plt.figure(figsize=(11, 5))
plt.plot(monthly["month"], monthly["revenue"], marker="o", linewidth=2, color="#2A5C8A")
plt.xticks(rotation=45, ha="right")
plt.title("Monthly Revenue Trend (2024–2025)", fontsize=14, fontweight="bold")
plt.ylabel("Revenue (₹)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig(OUT_DIR / "monthly_revenue_trend.png", dpi=150)
plt.close()

# ── 3. Revenue by Category ────────────────────────────────────────
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=cat_rev.values, y=cat_rev.index, palette="crest")
plt.title("Total Revenue by Category", fontsize=14, fontweight="bold")
plt.xlabel("Revenue (₹)")
plt.tight_layout()
plt.savefig(OUT_DIR / "revenue_by_category.png", dpi=150)
plt.close()

# ── 4. Revenue by Region ──────────────────────────────────────────
region_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

plt.figure(figsize=(7, 5))
sns.barplot(x=region_rev.index, y=region_rev.values, palette="flare")
plt.title("Total Revenue by Region", fontsize=14, fontweight="bold")
plt.ylabel("Revenue (₹)")
plt.tight_layout()
plt.savefig(OUT_DIR / "revenue_by_region.png", dpi=150)
plt.close()

print(f"\nCharts and KPI report saved to: {OUT_DIR}")
