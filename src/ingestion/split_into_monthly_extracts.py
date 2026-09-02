"""Slices the single UCI dataset into per-month CSV files.

The PDF's suggested-source table calls for "CSV/Excel order files -
simulated monthly extracts" as an incremental-loading source. Rather than
fabricating a second dataset, this script slices the UCI Online Retail
data (which already spans ~13 months) by invoice month, so the incremental
loader has real monthly files to consume one at a time.
"""
import pandas as pd

from src.ingestion.download_dataset import DEST_FILE as ONLINE_RETAIL_FILE
from src.utils.config import DATA_EXTERNAL_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MONTHLY_DIR = DATA_EXTERNAL_DIR / "monthly"


def split(source_file=ONLINE_RETAIL_FILE, out_dir=MONTHLY_DIR) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(source_file)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["_year_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    written = []
    for year_month, group in df.groupby("_year_month"):
        out_path = out_dir / f"{year_month}.csv"
        group.drop(columns=["_year_month"]).to_csv(out_path, index=False)
        written.append(out_path)
        logger.info("Wrote %d rows to %s", len(group), out_path)

    return sorted(written)


if __name__ == "__main__":
    split()
