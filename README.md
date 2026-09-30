# RetailPulse — Sales Trend Analysis & Customer Segmentation

An end-to-end data analytics pipeline that turns raw retail transaction
logs into KPI reporting, trend visualizations, and ML-driven customer
segments — built to mirror the kind of analysis a retail/e-commerce
analytics team runs monthly.

## What it does

1. **`generate_data.py`** — Generates a realistic 2-year synthetic retail
   transaction dataset (15,000 transactions, 800 customers, 5 regions,
   5 product categories) with seasonal demand patterns (holiday spike in
   Nov–Dec, mid-year dip) and customer spend tiers, so the rest of the
   pipeline has representative data to work with.

2. **`eda_trends.py`** — Exploratory data analysis and KPI reporting:
   - Computes headline KPIs: total revenue, transaction count, unique
     customers, average order value, YoY growth
   - Monthly revenue trend chart (seasonality clearly visible)
   - Revenue breakdown by category and by region
   - Exports a `kpi_summary.csv` suitable for feeding into a BI dashboard

3. **`customer_segmentation.py`** — RFM (Recency, Frequency, Monetary)
   feature engineering + K-Means clustering:
   - Builds RFM features per customer
   - Standardizes features and uses the elbow method + silhouette score
     to select the number of clusters (constrained to a business-
     interpretable range of 3–5 segments)
   - Labels clusters by RFM profile: **High-Value Active**,
     **Occasional/New**, **At-Risk/Dormant**
   - Outputs a segment summary and visualizations (scatter + pie chart)

## Key results (from the generated dataset)

| Segment | Customers | Avg. Recency (days) | Avg. Frequency | Avg. Monetary (₹) |
|---|---|---|---|---|
| High-Value Active | 115 | 9.4 | 46.2 | 666,795 |
| Occasional / New | 594 | 23.5 | 15.0 | 105,149 |
| At-Risk / Dormant | 91 | 156.3 | 8.8 | 51,451 |

**Business takeaway:** ~14% of customers (High-Value Active) drive a
disproportionate share of revenue and should be prioritized for loyalty
programs, while the At-Risk segment (11% of customers, inactive ~5 months
on average) is a clear target for a win-back campaign.

## Tech stack

- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn (K-Means, StandardScaler, silhouette
  score)
- **Visualization:** matplotlib, seaborn

## How to run

```bash
pip install -r requirements.txt
python generate_data.py          # creates data/retail_transactions.csv
python eda_trends.py             # creates KPI report + trend charts
python customer_segmentation.py  # creates RFM segments + charts
streamlit run app.py             # launches the interactive dashboard
```

All static outputs (CSVs + PNGs) are written to `outputs/`. The Streamlit
app (`app.py`) recomputes KPIs and RFM segments live and lets you filter
by date range, region, and category.

## Live dashboard

Deployed on Streamlit Community Cloud: **[add your deployed URL here once live]**

## Project structure

```
retail-pulse/
├── app.py
├── generate_data.py
├── eda_trends.py
├── customer_segmentation.py
├── requirements.txt
├── data/
│   └── retail_transactions.csv
└── outputs/
    ├── kpi_summary.csv
    ├── monthly_revenue_trend.png
    ├── revenue_by_category.png
    ├── revenue_by_region.png
    ├── segmentation_k_selection.png
    ├── customer_segments.csv
    ├── segment_summary.csv
    ├── customer_segments_scatter.png
    └── segment_distribution_pie.png
```

## Note on the data

The transaction data is synthetically generated (see `generate_data.py`
for the exact logic) rather than sourced from a real retailer, so the
project is fully reproducible and shareable without any data-licensing
concerns. The generation logic itself — seasonal demand curves, category
revenue mix, and customer spend tiers — reflects patterns commonly seen
in real retail transaction data, so the downstream analysis and
segmentation techniques are directly transferable to a real dataset.
