"""Computes recency/frequency/monetary features and a rule-based RFM segment
per customer, then upserts them into warehouse.fact_customer_retention.
"""
import pandas as pd

from src.db.connection import get_engine
from src.db.upsert import read_sql_df, upsert
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

QUERY = """
    SELECT f.customer_key, d.full_date, f.invoice_no, f.revenue
    FROM warehouse.fact_sales f
    JOIN warehouse.dim_date d ON f.date_key = d.date_key
    WHERE f.customer_key IS NOT NULL
"""


def _segment(score: int) -> str:
    if score >= 9:
        return "Champions"
    if score >= 7:
        return "Loyal"
    if score >= 5:
        return "At Risk"
    return "Lost"


def compute_rfm(engine, snapshot_date=None) -> pd.DataFrame:
    df = read_sql_df(engine, QUERY)
    if df.empty:
        return pd.DataFrame(columns=["customer_key", "snapshot_date", "recency_days", "frequency", "monetary", "rfm_segment"])
    df["full_date"] = pd.to_datetime(df["full_date"])

    snapshot_date = pd.Timestamp(snapshot_date) if snapshot_date else df["full_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_key").agg(
        last_purchase=("full_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()
    rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days

    # Quartile-score each dimension 1 (worst) - 4 (best); recency is inverted
    # (fewer days since last purchase = better score).
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    rfm["rfm_segment"] = (rfm["r_score"] + rfm["f_score"] + rfm["m_score"]).apply(_segment)

    rfm["snapshot_date"] = snapshot_date.date()
    return rfm[["customer_key", "snapshot_date", "recency_days", "frequency", "monetary", "rfm_segment"]]


def run(snapshot_date=None) -> dict:
    engine = get_engine()
    rfm_df = compute_rfm(engine, snapshot_date)
    upsert(
        engine, "warehouse", "fact_customer_retention", rfm_df,
        conflict_cols=["customer_key", "snapshot_date"],
        update_cols=["recency_days", "frequency", "monetary", "rfm_segment"],
    )
    summary = {"customers_scored": len(rfm_df)}
    logger.info("rfm_features summary: %s", summary)
    return summary


if __name__ == "__main__":
    run()
