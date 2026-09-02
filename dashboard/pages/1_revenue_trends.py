import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.charts import query  # noqa: E402

st.title("Revenue & Order Trends")

daily = query("""
    SELECT full_date, SUM(revenue) AS revenue, COUNT(DISTINCT stock_code) AS line_items
    FROM marts.sales_mart
    GROUP BY full_date
    ORDER BY full_date
""")

fig_rev = px.line(daily, x="full_date", y="revenue", title="Daily Revenue")
st.plotly_chart(fig_rev, use_container_width=True)

monthly = query("""
    SELECT year, month, month_name,
           SUM(revenue) AS revenue,
           COUNT(DISTINCT (country, full_date, stock_code)) AS line_items
    FROM marts.sales_mart
    GROUP BY year, month, month_name
    ORDER BY year, month
""")
monthly["period"] = monthly["month_name"].str[:3] + " " + monthly["year"].astype(str)
fig_orders = px.bar(monthly, x="period", y="revenue", title="Monthly Revenue")
st.plotly_chart(fig_orders, use_container_width=True)
