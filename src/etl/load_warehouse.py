"""Loads the cleaned staging CSV into the Postgres star schema.

Idempotent by design: dimension upserts use ON CONFLICT DO NOTHING and
fact_sales upserts on (invoice_no, stock_code) DO UPDATE, so re-running
this over the same (growing) staging file never creates duplicates.
"""
from pathlib import Path

import pandas as pd

from src.db.connection import get_engine
from src.db.upsert import read_sql_df, upsert as _upsert
from src.etl.clean_transform import CLEANED_SALES_FILE
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _build_dim_date(dates: pd.Series) -> pd.DataFrame:
    unique_dates = pd.to_datetime(dates.dropna().unique())
    df = pd.DataFrame({"full_date": unique_dates})
    df["date_key"] = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["day"] = df["full_date"].dt.day
    df["month"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.month_name()
    df["quarter"] = df["full_date"].dt.quarter
    df["year"] = df["full_date"].dt.year
    df["day_of_week"] = df["full_date"].dt.day_name()
    df["is_weekend"] = df["full_date"].dt.dayofweek.isin([5, 6])
    return df[["date_key", "full_date", "day", "month", "month_name", "quarter", "year", "day_of_week", "is_weekend"]]


def run(cleaned_file: Path = CLEANED_SALES_FILE) -> dict:
    engine = get_engine()
    df = pd.read_csv(
        cleaned_file,
        parse_dates=["invoice_date"],
        dtype={"invoice_no": str, "stock_code": str, "customer_id": str, "country": str},
    )
    df = df.drop_duplicates(subset=["invoice_no", "stock_code"], keep="last")

    # --- Dimensions ---
    dim_date_df = _build_dim_date(df["invoice_date"])
    _upsert(engine, "warehouse", "dim_date", dim_date_df, conflict_cols=["date_key"])

    dim_geography_df = pd.DataFrame({"country": df["country"].dropna().unique()})
    _upsert(engine, "warehouse", "dim_geography", dim_geography_df, conflict_cols=["country"])

    dim_product_df = (
        df[["stock_code", "description"]]
        .dropna(subset=["stock_code"])
        .drop_duplicates(subset=["stock_code"], keep="first")
    )
    _upsert(engine, "warehouse", "dim_product", dim_product_df, conflict_cols=["stock_code"])

    customers = df.dropna(subset=["customer_id"]).copy()
    dim_customer_df = (
        customers.groupby("customer_id")
        .agg(country=("country", "first"), first_purchase_date=("invoice_date", "min"))
        .reset_index()
    )
    dim_customer_df["first_purchase_date"] = dim_customer_df["first_purchase_date"].dt.date
    _upsert(engine, "warehouse", "dim_customer", dim_customer_df, conflict_cols=["customer_id"])

    # --- Fact table ---
    customer_map = read_sql_df(engine, "SELECT customer_id, customer_key FROM warehouse.dim_customer")
    product_map = read_sql_df(engine, "SELECT stock_code, product_key FROM warehouse.dim_product")
    geography_map = read_sql_df(engine, "SELECT country, geography_key FROM warehouse.dim_geography")

    fact_df = df.copy()
    fact_df["date_key"] = fact_df["invoice_date"].dt.strftime("%Y%m%d").astype(int)
    fact_df["revenue"] = (fact_df["quantity"] * fact_df["unit_price"]).round(2)
    fact_df = fact_df.merge(customer_map, on="customer_id", how="left")
    fact_df = fact_df.merge(product_map, on="stock_code", how="left")
    fact_df = fact_df.merge(geography_map, on="country", how="left")

    fact_cols = [
        "invoice_no", "stock_code", "date_key", "customer_key", "product_key",
        "geography_key", "quantity", "unit_price", "revenue",
    ]
    fact_sales_df = fact_df[fact_cols].dropna(subset=["product_key", "geography_key", "date_key"])
    fact_sales_df = fact_sales_df.astype({
        "date_key": int, "product_key": int, "geography_key": int, "quantity": int,
    })
    fact_sales_df["customer_key"] = fact_sales_df["customer_key"].astype("Int64")

    _upsert(
        engine, "warehouse", "fact_sales", fact_sales_df,
        conflict_cols=["invoice_no", "stock_code"],
        update_cols=["date_key", "customer_key", "product_key", "geography_key", "quantity", "unit_price", "revenue"],
    )

    summary = {
        "rows_processed": len(df),
        "fact_rows_loaded": len(fact_sales_df),
        "unique_customers": len(dim_customer_df),
        "unique_products": len(dim_product_df),
    }
    logger.info("load_warehouse summary: %s", summary)
    return summary


if __name__ == "__main__":
    run()
