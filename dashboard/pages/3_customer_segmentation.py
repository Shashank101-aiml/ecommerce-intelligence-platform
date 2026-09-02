import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.charts import query  # noqa: E402

st.title("Customer Segmentation (RFM)")

rfm = query("""
    SELECT customer_id, country, recency_days, frequency, monetary, rfm_segment
    FROM marts.customer_mart
    WHERE rfm_segment IS NOT NULL
""")

if rfm.empty:
    st.warning("No RFM data yet — run the rfm_features step of the pipeline first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    seg_counts = rfm["rfm_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]
    fig_pie = px.pie(seg_counts, names="segment", values="customers", title="Customers by RFM Segment")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_scatter = px.scatter(
        rfm, x="recency_days", y="monetary", color="rfm_segment", size="frequency",
        hover_data=["customer_id", "country"], title="Recency vs Monetary Value",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.dataframe(rfm.sort_values("monetary", ascending=False), use_container_width=True)
