"""
generate_data.py — Synthetic Retail Transactions Dataset Generator
====================================================================
Generates a realistic 2-year retail transaction log with seasonal
sales patterns, category/region mix, and customer purchase behaviour,
so the rest of the pipeline (EDA + segmentation) has something
representative to work with.

NOTE: This is a synthetically generated dataset (not scraped from a
real retailer) — built this way so the project is fully reproducible
and shareable without any data-licensing concerns. The generation
logic itself (seasonality, category skew, customer tiers) reflects
patterns commonly seen in real retail transaction logs.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_CUSTOMERS = 800
N_TRANSACTIONS = 15000
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.28, 0.22, 0.18, 0.20, 0.12]

CATEGORIES = {
    "Electronics":   (2500, 45000, 0.20),
    "Apparel":       (400,  4500,  0.28),
    "Home & Living":  (600,  8000,  0.18),
    "Grocery":       (50,   1200,  0.22),
    "Beauty":        (150,  3000,  0.12),
}
CAT_NAMES = list(CATEGORIES.keys())
CAT_WEIGHTS = [v[2] for v in CATEGORIES.values()]

# Assign each customer a tier that drives purchase frequency & spend
CUSTOMER_TIERS = np.random.choice(
    ["Low", "Medium", "High"], size=N_CUSTOMERS, p=[0.55, 0.32, 0.13]
)
TIER_FREQ_WEIGHT = {"Low": 0.5, "Medium": 1.0, "High": 2.2}
TIER_SPEND_MULT = {"Low": 0.7, "Medium": 1.0, "High": 1.8}

customer_ids = np.arange(1, N_CUSTOMERS + 1)
customer_region = np.random.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)

# Sampling weight per customer (drives how many transactions they get)
cust_weights = np.array([TIER_FREQ_WEIGHT[t] for t in CUSTOMER_TIERS])
cust_weights = cust_weights / cust_weights.sum()

chosen_customers = np.random.choice(customer_ids, size=N_TRANSACTIONS, p=cust_weights)

# Seasonal weighting: boost Nov-Dec (holiday season) and a mild mid-year dip
date_range = pd.date_range(START_DATE, END_DATE, freq="D")
month_boost = {11: 1.6, 12: 1.9, 1: 0.85, 6: 0.8, 7: 0.85}
day_weights = np.array([month_boost.get(d.month, 1.0) for d in date_range], dtype=float)
day_weights = day_weights / day_weights.sum()
transaction_dates = np.random.choice(date_range, size=N_TRANSACTIONS, p=day_weights)

rows = []
for i in range(N_TRANSACTIONS):
    cust_id = chosen_customers[i]
    tier = CUSTOMER_TIERS[cust_id - 1]
    region = customer_region[cust_id - 1]
    category = np.random.choice(CAT_NAMES, p=CAT_WEIGHTS)
    low, high, _ = CATEGORIES[category]

    unit_price = np.random.uniform(low, high) * TIER_SPEND_MULT[tier]
    quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.2, 0.15, 0.1, 0.06, 0.04])
    revenue = round(unit_price * quantity, 2)

    rows.append({
        "transaction_id": f"TXN{100000 + i}",
        "date": pd.Timestamp(transaction_dates[i]),
        "customer_id": f"CUST{cust_id:04d}",
        "region": region,
        "category": category,
        "quantity": quantity,
        "unit_price": round(unit_price, 2),
        "revenue": revenue,
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

out_path = DATA_DIR / "retail_transactions.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df):,} transactions across {N_CUSTOMERS} customers.")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total revenue: ₹{df['revenue'].sum():,.0f}")
print(f"Saved to: {out_path}")
