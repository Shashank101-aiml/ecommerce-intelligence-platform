import os
import sys
from pathlib import Path

import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components.charts import query  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIER_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}

st.title("Churn Risk")
st.caption("Probability that a customer makes no purchase in the next 90 days, from the registered champion model.")

# ---- model & monitoring status -------------------------------------------------------------
health = None
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
except requests.RequestException:
    pass

c1, c2, c3 = st.columns(3)
if health and health.get("model_loaded"):
    c1.metric("Prediction API", "Online")
    c2.metric("Champion model", f"v{health['model_version']}")
else:
    c1.metric("Prediction API", "Offline")
    c2.metric("Champion model", "unavailable")

status = query("""
    SELECT metric_name, feature_name, metric_value, breached, measured_at
    FROM warehouse.monitoring_metrics
    WHERE measured_at = (SELECT MAX(measured_at) FROM warehouse.monitoring_metrics)
""")
if status.empty:
    c3.metric("Model monitoring", "No data yet")
else:
    breaches = status[status["breached"]]
    c3.metric("Monitoring breaches", f"{len(breaches)} of {len(status)} checks")
    with st.expander(f"Monitoring details (measured {status['measured_at'].iloc[0]:%Y-%m-%d %H:%M})"):
        if breaches.empty:
            st.success("All checks within thresholds.")
        else:
            st.warning("Checks outside thresholds — see retraining policy in docs/model_lifecycle.md.")
            st.dataframe(breaches[["metric_name", "feature_name", "metric_value"]], use_container_width=True)

st.divider()

# ---- batch scores --------------------------------------------------------------------------
scores = query("""
    SELECT c.customer_id, c.country, s.churn_probability, s.risk_tier, s.as_of_date, s.model_version,
           r.recency_days, r.frequency, r.monetary, r.rfm_segment
    FROM warehouse.customer_churn_scores s
    JOIN warehouse.dim_customer c ON s.customer_key = c.customer_key
    LEFT JOIN marts.customer_mart r ON r.customer_key = s.customer_key
    WHERE s.as_of_date = (SELECT MAX(as_of_date) FROM warehouse.customer_churn_scores)
""")

if scores.empty:
    st.info("No batch scores yet — run `python -m ml.scoring.score_customers` or the ml_pipeline_dag.")
else:
    tiers = scores["risk_tier"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{len(scores):,}")
    k2.metric("High risk", f"{tiers.get('High', 0):,}")
    k3.metric("Revenue at high risk", f"£{scores.loc[scores['risk_tier'] == 'High', 'monetary'].astype(float).sum():,.0f}")
    k4.metric("Scored as of", str(scores["as_of_date"].iloc[0]))

    left, right = st.columns(2)
    with left:
        tier_df = tiers.reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        tier_df.columns = ["risk_tier", "customers"]
        st.plotly_chart(px.bar(tier_df, x="risk_tier", y="customers", color="risk_tier",
                               color_discrete_map=TIER_COLORS, title="Customers by risk tier"),
                        use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(scores, x="churn_probability", nbins=30, title="Churn probability distribution"),
                        use_container_width=True)

    seg = scores.dropna(subset=["rfm_segment"]).groupby(["rfm_segment", "risk_tier"]).size().reset_index(name="customers")
    st.plotly_chart(px.bar(seg, x="rfm_segment", y="customers", color="risk_tier", color_discrete_map=TIER_COLORS,
                           title="Risk tier within each RFM segment"), use_container_width=True)

    st.subheader("Highest-value customers at risk")
    top = scores[scores["risk_tier"] == "High"].sort_values("monetary", ascending=False).head(50)
    st.dataframe(top[["customer_id", "country", "churn_probability", "monetary", "frequency", "recency_days", "rfm_segment"]],
                 use_container_width=True)

st.divider()

# ---- live single-customer lookup through the API --------------------------------------------
st.subheader("Look up a customer (live prediction)")
customer_id = st.text_input("Customer ID", placeholder="e.g. 12346")
if customer_id:
    try:
        resp = requests.post(f"{API_URL}/predict/customer/{customer_id.strip()}", timeout=15)
        if resp.status_code == 200:
            body = resp.json()
            st.metric("Churn probability", f"{body['churn_probability']:.1%}",
                      "predicted to churn" if body["predicted_label"] else "predicted to stay", delta_color="inverse")
            st.caption(f"Model v{body['model_version']}, features as of {body['as_of_date']}")
        elif resp.status_code == 404:
            st.warning(f"Customer {customer_id} not found.")
        else:
            st.error(f"Prediction API returned HTTP {resp.status_code}.")
    except requests.RequestException:
        st.error("Prediction API is unreachable.")
