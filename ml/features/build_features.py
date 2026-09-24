"""Point-in-time churn features and labels.

For a cutoff date, features use only purchases on or before the cutoff; the
label looks at the CHURN_WINDOW_DAYS after it. A customer is `churned` (1) if
they were active before the cutoff and made no purchase in that window.
"""
from datetime import date

import pandas as pd

from ml.config import CHURN_WINDOW_DAYS, FEATURE_COLUMNS, LABEL_COLUMN
from src.db.connection import get_engine
from src.db.upsert import read_sql_df, upsert
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ORDERS_QUERY = """
    SELECT f.customer_key, d.full_date, f.invoice_no, f.product_key, f.revenue, g.country
    FROM warehouse.fact_sales f
    JOIN warehouse.dim_date d ON f.date_key = d.date_key
    JOIN warehouse.dim_geography g ON f.geography_key = g.geography_key
    WHERE f.customer_key IS NOT NULL
"""


def load_orders(engine) -> pd.DataFrame:
    df = read_sql_df(engine, ORDERS_QUERY)
    df["full_date"] = pd.to_datetime(df["full_date"])
    df["revenue"] = df["revenue"].astype(float)
    return df


def _window_stats(invoices: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    recent = invoices[invoices["date"] > cutoff - pd.Timedelta(days=days)]
    out = recent.groupby("customer_key").agg(**{f"orders_{days}d": ("invoice_no", "count"), f"revenue_{days}d": ("revenue", "sum")})
    return out


def build_features(orders: pd.DataFrame, cutoff: date, window_days: int = CHURN_WINDOW_DAYS,
                   with_label: bool = True) -> pd.DataFrame:
    """Features (and optionally the churn label) for every customer active on/before `cutoff`."""
    cutoff_ts = pd.Timestamp(cutoff)
    history = orders[orders["full_date"] <= cutoff_ts]
    if history.empty:
        return pd.DataFrame(columns=["customer_key", "cutoff_date", *FEATURE_COLUMNS, LABEL_COLUMN])

    invoices = history.groupby(["customer_key", "invoice_no"]).agg(
        date=("full_date", "min"), revenue=("revenue", "sum")).reset_index()

    base = invoices.groupby("customer_key").agg(
        first_purchase=("date", "min"), last_purchase=("date", "max"),
        frequency=("invoice_no", "count"), monetary=("revenue", "sum"))
    base["recency_days"] = (cutoff_ts - base["last_purchase"]).dt.days
    base["tenure_days"] = (cutoff_ts - base["first_purchase"]).dt.days
    base["avg_order_value"] = base["monetary"] / base["frequency"]

    span_days = (base["last_purchase"] - base["first_purchase"]).dt.days
    base["avg_days_between_orders"] = (span_days / (base["frequency"] - 1).where(base["frequency"] > 1)).fillna(0.0)

    for days in (30, 60, 90):
        base = base.join(_window_stats(invoices, cutoff_ts, days), how="left")
    base = base.fillna({c: 0 for c in ["orders_30d", "orders_60d", "orders_90d", "revenue_30d", "revenue_60d", "revenue_90d"]})

    base["distinct_products"] = history.groupby("customer_key")["product_key"].nunique()
    uk = history.groupby("customer_key")["country"].agg(lambda s: int((s == "United Kingdom").any()))
    base["is_uk"] = uk

    result = base.reset_index()
    result["cutoff_date"] = cutoff

    if with_label:
        window_end = cutoff_ts + pd.Timedelta(days=window_days)
        if window_end > orders["full_date"].max():
            raise ValueError(f"Label window ending {window_end.date()} extends past the last data date")
        future = orders[(orders["full_date"] > cutoff_ts) & (orders["full_date"] <= window_end)]
        active = set(future["customer_key"].unique())
        result[LABEL_COLUMN] = (~result["customer_key"].isin(active)).astype(int)
    else:
        result[LABEL_COLUMN] = None

    return result[["customer_key", "cutoff_date", *FEATURE_COLUMNS, LABEL_COLUMN]]


def run(cutoffs: list | None = None) -> dict:
    from ml.config import TEST_CUTOFFS, TRAIN_CUTOFFS, VALIDATION_CUTOFFS
    cutoffs = cutoffs or [*TRAIN_CUTOFFS, *VALIDATION_CUTOFFS, *TEST_CUTOFFS]
    engine = get_engine()
    orders = load_orders(engine)
    summary = {}
    for cutoff in cutoffs:
        feats = build_features(orders, cutoff)
        upsert(engine, "warehouse", "ml_customer_features", feats,
               conflict_cols=["customer_key", "cutoff_date"],
               update_cols=[*FEATURE_COLUMNS, LABEL_COLUMN])
        summary[str(cutoff)] = {"customers": len(feats), "churn_rate": round(float(feats[LABEL_COLUMN].mean()), 3)}
        logger.info("cutoff %s: %s", cutoff, summary[str(cutoff)])
    return summary


if __name__ == "__main__":
    run()
