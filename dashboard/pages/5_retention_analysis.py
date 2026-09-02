import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.charts import query  # noqa: E402

st.title("Retention & Repeat-Purchase Analysis")

frequency = query("""
    SELECT frequency
    FROM marts.customer_mart
    WHERE frequency IS NOT NULL
""")

if frequency.empty:
    st.warning("No retention data yet — run the rfm_features step of the pipeline first.")
    st.stop()

repeat_rate = (frequency["frequency"] > 1).mean() * 100
st.metric("Repeat-Purchase Rate", f"{repeat_rate:.1f}%")

fig_hist = px.histogram(
    frequency, x="frequency", nbins=20,
    title="Distribution of Orders per Customer",
)
st.plotly_chart(fig_hist, use_container_width=True)

segment_summary = query("""
    SELECT rfm_segment, COUNT(*) AS customers, AVG(monetary) AS avg_monetary, AVG(frequency) AS avg_frequency
    FROM marts.customer_mart
    WHERE rfm_segment IS NOT NULL
    GROUP BY rfm_segment
    ORDER BY avg_monetary DESC
""")
st.dataframe(segment_summary, use_container_width=True)
