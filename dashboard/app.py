import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.charts import query  # noqa: E402

st.set_page_config(page_title="E-Commerce Sales Analytics", layout="wide")

st.title("E-Commerce Sales Analytics")
st.caption("UCI Online Retail dataset — sales, customer, and retention overview")

try:
    kpis = query("""
        SELECT
            COUNT(DISTINCT invoice_no) AS orders,
            SUM(revenue) AS revenue,
            COUNT(DISTINCT customer_key) AS customers,
            MIN(full_date) AS start_date,
            MAX(full_date) AS end_date
        FROM marts.sales_mart
    """).iloc[0]
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the warehouse database: {exc}")
    st.info("Run the Airflow DAG (or the ETL scripts directly) first, then reload this page.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{kpis['revenue']:,.0f}")
col2.metric("Total Orders", f"{kpis['orders']:,}")
col3.metric("Unique Customers", f"{kpis['customers']:,}")
col4.metric("Date Range", f"{kpis['start_date']} → {kpis['end_date']}")

st.divider()

monthly = query("""
    SELECT year, month, month_name, SUM(revenue) AS revenue
    FROM marts.sales_mart
    GROUP BY year, month, month_name
    ORDER BY year, month
""")
monthly["period"] = monthly["month_name"].str[:3] + " " + monthly["year"].astype(str)
fig = px.line(monthly, x="period", y="revenue", markers=True, title="Monthly Revenue Trend")
st.plotly_chart(fig, use_container_width=True)

st.info("Use the pages in the sidebar for product, customer, geography, and retention views.")
