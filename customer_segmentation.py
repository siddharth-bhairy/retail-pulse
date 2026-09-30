"""
customer_segmentation.py — RFM Feature Engineering + K-Means Segmentation
============================================================================
Builds Recency, Frequency, Monetary (RFM) features per customer from the
transaction log, then applies K-Means clustering to segment customers
into actionable groups (e.g. High-Value Loyalists, At-Risk, New/Low-Spend).

Pipeline:
  1. Compute RFM features per customer.
  2. Standardise features (K-Means is distance-based).
  3. Use the elbow method to pick a sensible k.
  4. Fit K-Means, label clusters by their RFM profile.
  5. Save segment assignments + a segment-profile summary + charts.

Usage:
    python customer_segmentation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retail_transactions.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
snapshot_date = df["date"].max() + pd.Timedelta(days=1)

# ── 1. RFM Feature Engineering ────────────────────────────────────
rfm = df.groupby("customer_id").agg(
    recency=("date", lambda x: (snapshot_date - x.max()).days),
    frequency=("transaction_id", "count"),
    monetary=("revenue", "sum"),
).reset_index()

# ── 2. Standardise ────────────────────────────────────────────────
features = rfm[["recency", "frequency", "monetary"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# ── 3. Elbow method to choose k ───────────────────────────────────
inertias, sil_scores, k_range = [], [], range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled, labels))

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(list(k_range), inertias, marker="o", color="#2A5C8A")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(list(k_range), sil_scores, marker="o", color="#C9562A")
plt.title("Silhouette Score by k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "segmentation_k_selection.png", dpi=150)
plt.close()

# Prefer k in [3,5]: 2 clusters is statistically "cleanest" here but too
# coarse to be actionable for a marketing/CRM team, so we pick the best
# silhouette score within a business-interpretable range instead.
candidate_ks = [k for k in k_range if 3 <= k <= 5]
candidate_scores = [sil_scores[list(k_range).index(k)] for k in candidate_ks]
best_k = candidate_ks[int(np.argmax(candidate_scores))]
print(f"Selected k={best_k} (silhouette = {max(candidate_scores):.3f}, "
      f"chosen from k=3-5 range for business interpretability)")

# ── 4. Fit final K-Means model ────────────────────────────────────
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
rfm["cluster"] = kmeans.fit_predict(scaled)

# Label clusters by RFM profile relative to overall medians
cluster_profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
r_med, f_med, m_med = rfm["recency"].median(), rfm["frequency"].median(), rfm["monetary"].median()

def label_cluster(row):
    if row["monetary"] >= m_med and row["recency"] <= r_med:
        return "High-Value Active"
    if row["recency"] > r_med and row["frequency"] <= f_med:
        return "At-Risk / Dormant"
    if row["frequency"] >= f_med and row["monetary"] < m_med:
        return "Frequent, Low-Spend"
    return "Occasional / New"

cluster_profile["segment_label"] = cluster_profile.apply(label_cluster, axis=1)
label_map = cluster_profile["segment_label"].to_dict()
rfm["segment"] = rfm["cluster"].map(label_map)

rfm.to_csv(OUT_DIR / "customer_segments.csv", index=False)

segment_summary = rfm.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_recency_days=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
).round(1).sort_values("avg_monetary", ascending=False)
segment_summary.to_csv(OUT_DIR / "segment_summary.csv")

print("\n── Segment Summary ──")
print(segment_summary.to_string())

# ── 5. Visualise segments ─────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=rfm, x="recency", y="monetary", hue="segment",
    palette="Set2", s=45, alpha=0.75,
)
plt.title("Customer Segments — Recency vs Monetary Value", fontsize=13, fontweight="bold")
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Monetary Value (₹)")
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "customer_segments_scatter.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
segment_counts = rfm["segment"].value_counts()
plt.pie(
    segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%",
    colors=sns.color_palette("Set2"), startangle=90,
)
plt.title("Customer Segment Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "segment_distribution_pie.png", dpi=150)
plt.close()

print(f"\nSegmentation outputs saved to: {OUT_DIR}")
