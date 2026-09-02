"""Copies a source file into the immutable raw layer and logs extraction metadata.

Satisfies the PDF's minimum ingestion expectations: maintain a raw copy,
record extraction date/source/status/row count, and handle+log failures
without ever touching the final analytical dataset.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.config import DATA_RAW_DIR, EXTRACTION_LOG_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
EXTRACTION_LOG_FILE = EXTRACTION_LOG_DIR / "extraction_log.jsonl"


def _read_row_count(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        return sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
    return len(pd.read_excel(path))


def _append_log(entry: dict) -> None:
    EXTRACTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRACTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def ingest_file(source_path: Path, source_name: str) -> dict:
    """Copies source_path into data/raw/ and logs the extraction attempt.

    Returns the log entry dict, including the raw copy's path on success
    (or None on failure).
    """
    source_path = Path(source_path)
    extraction_date = datetime.now(timezone.utc).isoformat()
    entry = {
        "extraction_date": extraction_date,
        "source": source_name,
        "source_file": str(source_path),
        "status": "failed",
        "row_count": 0,
        "raw_copy": None,
        "error": None,
    }

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        row_count = _read_row_count(source_path)

        run_dir = DATA_RAW_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_copy = run_dir / source_path.name
        shutil.copy2(source_path, raw_copy)

        entry.update(status="success", row_count=row_count, raw_copy=str(raw_copy))
        logger.info("Ingested %s -> %s (%d rows)", source_path, raw_copy, row_count)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: log and move on
        entry["error"] = str(exc)
        logger.error("Failed to ingest %s: %s", source_path, exc)

    _append_log(entry)
    return entry


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m src.ingestion.ingest_raw <source_path> <source_name>")
        sys.exit(1)
    ingest_file(Path(sys.argv[1]), sys.argv[2])
