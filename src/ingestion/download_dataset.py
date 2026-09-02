"""Downloads the UCI Online Retail dataset (the PDF's required primary source)."""
import requests

from src.utils.config import DATA_EXTERNAL_DIR, UCI_ONLINE_RETAIL_URL
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEST_FILE = DATA_EXTERNAL_DIR / "online_retail.xlsx"


def download(url: str = UCI_ONLINE_RETAIL_URL, dest=DEST_FILE, force: bool = False):
    if dest.exists() and not force:
        logger.info("Dataset already present at %s, skipping download", dest)
        return dest

    logger.info("Downloading UCI Online Retail dataset from %s", url)
    response = requests.get(url, timeout=120, headers={"User-Agent": "ecommerce-analytics-mlops/1.0"})
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info("Saved dataset to %s (%d bytes)", dest, len(response.content))
    return dest


if __name__ == "__main__":
    download()
