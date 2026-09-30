"""
app.py — RetailPulse Interactive Dashboard
=============================================
Streamlit dashboard for the RetailPulse sales-trend and customer-
segmentation analysis. Loads the transaction dataset, lets the viewer
filter by date range / region / category, and shows:

  - KPI cards (revenue, transactions, AOV, YoY growth)
  - Monthly revenue trend
  - Revenue by category / region
  - RFM-based customer segmentation (K-Means), computed live and
    cached, with a segment scatter plot and profile summary

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then deploy at share.streamlit.io
(Community Cloud) pointing at this file.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retail_transactions.csv"
RANDOM_SEED = 42


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    return df


@st.cache_data
def compute_segments(df: pd.DataFrame):
    """RFM feature engineering + K-Means segmentation (mirrors
    customer_segmentation.py), cached so filtering the rest of the
    dashboard doesn't retrigger clustering."""
    snapshot_date = df["date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("date", lambda x: (snapshot_date - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary=("revenue", "sum"),
    ).reset_index()

    scaled = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

    k_range = range(3, 6)
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(scaled)
        sil_scores.append(silhouette_score(scaled, labels))
    best_k = list(k_range)[int(np.argmax(sil_scores))]

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    r_med, f_med, m_med = rfm["recency"].median(), rfm["frequency"].median(), rfm["monetary"].median()

    def label_cluster(row):
        if row["monetary"] >= m_med and row["recency"] <= r_med:
            return "High-Value Active"
        if row["recency"] > r_med and row["frequency"] <= f_med:
            return "At-Risk / Dormant"
        if row["frequency"] >= f_med and row["monetary"] < m_med:
            return "Frequent, Low-Spend"
        return "Occasional / New"

    profile["segment"] = profile.apply(label_cluster, axis=1)
    rfm["segment"] = rfm["cluster"].map(profile["segment"].to_dict())
    return rfm


df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────
st.sidebar.header("Filters")
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
categories = st.sidebar.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))

if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask = (df["date"] >= start) & (df["date"] <= end) & df["region"].isin(regions) & df["category"].isin(categories)
    filtered = df[mask]
else:
    filtered = df[df["region"].isin(regions) & df["category"].isin(categories)]

st.title("📊 RetailPulse — Sales Trend & Customer Segmentation")
st.caption("Interactive dashboard over synthetic retail transaction data. Use the sidebar to filter.")

# ── KPI row ────────────────────────────────────────────────────────
total_revenue = filtered["revenue"].sum()
total_txns = len(filtered)
aov = total_revenue / total_txns if total_txns else 0
unique_customers = filtered["customer_id"].nunique()

yearly = filtered.groupby("year")["revenue"].sum()
yoy = (yearly.iloc[-1] / yearly.iloc[0] - 1) * 100 if len(yearly) >= 2 and yearly.iloc[0] else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Transactions", f"{total_txns:,}")
c3.metric("Unique Customers", f"{unique_customers:,}")
c4.metric("Avg. Order Value", f"₹{aov:,.0f}")
c5.metric("YoY Revenue Growth", f"{yoy:.1f}%" if yoy is not None else "N/A")

st.divider()

# ── Trend + breakdowns ────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    monthly = filtered.groupby("month")["revenue"].sum().reset_index()
    fig = px.line(monthly, x="month", y="revenue", markers=True, title="Monthly Revenue Trend")
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cat_rev = filtered.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    fig2 = px.bar(cat_rev, x="revenue", y="category", orientation="h", title="Revenue by Category")
    fig2.update_layout(yaxis_title="", xaxis_title="Revenue (₹)")
    st.plotly_chart(fig2, use_container_width=True)

region_rev = filtered.groupby("region")["revenue"].sum().sort_values(ascending=False).reset_index()
fig3 = px.bar(region_rev, x="region", y="revenue", title="Revenue by Region")
fig3.update_layout(xaxis_title="Region", yaxis_title="Revenue (₹)")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Customer segmentation ─────────────────────────────────────────
st.subheader("🧩 Customer Segmentation (RFM + K-Means)")
st.caption("Computed on the full customer base (unaffected by the region/category filters above, "
           "since RFM needs each customer's complete purchase history).")

rfm = compute_segments(df)

seg_col1, seg_col2 = st.columns([2, 1])

with seg_col1:
    fig4 = px.scatter(
        rfm, x="recency", y="monetary", color="segment",
        hover_data=["customer_id", "frequency"],
        title="Customers — Recency vs Monetary Value",
        labels={"recency": "Recency (days since last purchase)", "monetary": "Monetary Value (₹)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

with seg_col2:
    seg_counts = rfm["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]
    fig5 = px.pie(seg_counts, names="segment", values="customers", title="Segment Distribution")
    st.plotly_chart(fig5, use_container_width=True)

summary = rfm.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_recency_days=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
).round(1).sort_values("avg_monetary", ascending=False)
st.dataframe(summary, use_container_width=True)

st.caption("Built with Streamlit + Plotly. Source: github.com/siddharth-bhairy — RetailPulse")
