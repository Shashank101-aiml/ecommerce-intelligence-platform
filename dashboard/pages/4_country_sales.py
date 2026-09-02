import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.charts import query  # noqa: E402

st.title("Country-wise Sales")

by_country = query("""
    SELECT country, SUM(revenue) AS revenue, COUNT(DISTINCT stock_code) AS distinct_products
    FROM marts.sales_mart
    GROUP BY country
    ORDER BY revenue DESC
""")

fig_map = px.choropleth(
    by_country, locations="country", locationmode="country names",
    color="revenue", title="Revenue by Country",
)
st.plotly_chart(fig_map, use_container_width=True)

fig_bar = px.bar(
    by_country.head(15).sort_values("revenue"),
    x="revenue", y="country", orientation="h", title="Top 15 Countries by Revenue",
)
st.plotly_chart(fig_bar, use_container_width=True)
