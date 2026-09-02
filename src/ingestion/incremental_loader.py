"""Feeds the simulated monthly extracts into the raw layer one at a time.

Tracks which monthly files have already been ingested in
logs/extraction_log/processed_files.json, so re-running this script only
picks up new/unseen monthly extracts (the "repeatable/scheduled ingestion
workflow" requirement).
"""
import json
from pathlib import Path

from src.ingestion.ingest_raw import ingest_file
from src.ingestion.split_into_monthly_extracts import MONTHLY_DIR
from src.utils.config import EXTRACTION_LOG_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
PROCESSED_FILE = EXTRACTION_LOG_DIR / "processed_files.json"


def _load_processed() -> set:
    if PROCESSED_FILE.exists():
        return set(json.loads(PROCESSED_FILE.read_text()))
    return set()


def _save_processed(processed: set) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.write_text(json.dumps(sorted(processed), indent=2))


def run_incremental_ingestion(monthly_dir: Path = MONTHLY_DIR) -> list:
    if not monthly_dir.exists() or not any(monthly_dir.glob("*.csv")):
        logger.warning(
            "No monthly extracts found in %s. Run split_into_monthly_extracts.py first.",
            monthly_dir,
        )
        return []

    processed = _load_processed()
    new_entries = []

    for monthly_file in sorted(monthly_dir.glob("*.csv")):
        if monthly_file.name in processed:
            continue
        entry = ingest_file(monthly_file, source_name=f"monthly_extract:{monthly_file.stem}")
        if entry["status"] == "success":
            processed.add(monthly_file.name)
        new_entries.append(entry)

    _save_processed(processed)
    logger.info("Incremental ingestion complete: %d new file(s) processed", len(new_entries))
    return new_entries


if __name__ == "__main__":
    run_incremental_ingestion()
