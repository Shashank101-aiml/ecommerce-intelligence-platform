"""Cleans and standardizes raw sales extracts into the staging layer.

Pipeline stage: raw (data/raw/**) -> staging (data/staging/cleaned_sales.csv
+ staging.sales_raw / staging.rejected_records in Postgres).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.db.connection import get_engine
from src.db.upsert import bulk_insert
from src.etl.data_quality_checks import validate_dataframe
from src.utils.config import DATA_REJECTED_DIR, DATA_STAGING_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CLEANED_SALES_FILE = DATA_STAGING_DIR / "cleaned_sales.csv"

COLUMN_MAP = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def _load_source_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["source_file"] = str(path)
    return df


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].apply(
        lambda v: str(int(v)) if pd.notna(v) else None
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").round(2)
    return df


def run(file_paths: list[Path], batch_id: str | None = None) -> dict:
    batch_id = batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_frames = [_load_source_file(Path(p)) for p in file_paths]
    combined = pd.concat(raw_frames, ignore_index=True)
    standardized = _standardize(combined)

    engine = get_engine()
    raw_cols = [c for c in COLUMN_MAP.values()] + ["source_file"]
    staging_df = standardized[raw_cols].copy()
    staging_df["batch_id"] = batch_id
    bulk_insert(engine, "staging", "sales_raw", staging_df)

    valid_df, rejected_df = validate_dataframe(standardized)

    if not rejected_df.empty:
        rejected_path = DATA_REJECTED_DIR / f"rejected_{batch_id}.csv"
        rejected_df.to_csv(rejected_path, index=False)

        rejected_records = pd.DataFrame({
            "raw_data": rejected_df.drop(columns=["rejection_reason"]).apply(
                lambda r: json.dumps(
                    {k: (None if pd.isna(v) else v) for k, v in r.items()}, default=str
                ),
                axis=1,
            ),
            "reason": rejected_df["rejection_reason"],
            "source_file": rejected_df["source_file"],
            "batch_id": batch_id,
        })
        bulk_insert(engine, "staging", "rejected_records", rejected_records)
        logger.warning("Rejected %d rows in batch %s (see %s)", len(rejected_df), batch_id, rejected_path)

    output_cols = [
        "invoice_no", "stock_code", "description", "quantity",
        "invoice_date", "unit_price", "customer_id", "country",
    ]
    write_header = not CLEANED_SALES_FILE.exists()
    valid_df[output_cols].to_csv(CLEANED_SALES_FILE, mode="a", index=False, header=write_header)

    summary = {
        "batch_id": batch_id,
        "input_rows": len(standardized),
        "valid_rows": len(valid_df),
        "rejected_rows": len(rejected_df),
    }
    logger.info("clean_transform summary: %s", summary)
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.etl.clean_transform <file1> [file2 ...]")
        sys.exit(1)
    run([Path(p) for p in sys.argv[1:]])
