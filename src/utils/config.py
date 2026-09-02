"""Central place for reading environment configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_EXTERNAL_DIR = Path(os.getenv("DATA_EXTERNAL_DIR", PROJECT_ROOT / "data" / "external"))
DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
DATA_STAGING_DIR = Path(os.getenv("DATA_STAGING_DIR", PROJECT_ROOT / "data" / "staging"))
DATA_REJECTED_DIR = Path(os.getenv("DATA_REJECTED_DIR", PROJECT_ROOT / "data" / "rejected"))
EXTRACTION_LOG_DIR = Path(os.getenv("EXTRACTION_LOG_DIR", PROJECT_ROOT / "logs" / "extraction_log"))

UCI_ONLINE_RETAIL_URL = os.getenv(
    "UCI_ONLINE_RETAIL_URL",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
)


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ecommerce")
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


for _dir in (DATA_EXTERNAL_DIR, DATA_RAW_DIR, DATA_STAGING_DIR, DATA_REJECTED_DIR, EXTRACTION_LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
