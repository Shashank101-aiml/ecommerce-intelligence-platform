import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.charts import query  # noqa: E402

st.title("Top Products & Categories")

top_n = st.slider("Number of top products", 5, 30, 10)

top_products = query(f"""
    SELECT stock_code, description, SUM(revenue) AS revenue, SUM(quantity) AS units_sold
    FROM marts.sales_mart
    GROUP BY stock_code, description
    ORDER BY revenue DESC
    LIMIT {top_n}
""")

fig = px.bar(
    top_products.sort_values("revenue"),
    x="revenue", y="description", orientation="h",
    title=f"Top {top_n} Products by Revenue",
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(top_products, use_container_width=True)
