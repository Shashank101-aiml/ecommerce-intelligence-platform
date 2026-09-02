"""End-to-end retail ETL DAG: ingest -> clean/validate -> load warehouse -> RFM features.

Runs against the simulated monthly extracts so each scheduled/triggered run
demonstrates incremental ingestion, not a single one-shot bulk load.
"""
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/project")

from src.ingestion.download_dataset import download  # noqa: E402
from src.ingestion.split_into_monthly_extracts import split  # noqa: E402
from src.ingestion.incremental_loader import run_incremental_ingestion  # noqa: E402
from src.etl.clean_transform import run as run_clean_transform  # noqa: E402
from src.etl.load_warehouse import run as run_load_warehouse  # noqa: E402
from src.features.rfm_features import run as run_rfm_features  # noqa: E402

default_args = {
    "owner": "student",
    "retries": 1,
}


def _download_and_split(**_):
    path = download()
    split(source_file=path)


def _incremental_ingest(**context):
    entries = run_incremental_ingestion()
    successful = [e["raw_copy"] for e in entries if e["status"] == "success"]
    context["ti"].xcom_push(key="raw_files", value=successful)


def _clean(**context):
    raw_files = context["ti"].xcom_pull(key="raw_files", task_ids="incremental_ingest")
    if not raw_files:
        return
    run_clean_transform(raw_files)


def _load(**_):
    run_load_warehouse()


def _features(**_):
    run_rfm_features()


with DAG(
    dag_id="ecommerce_etl_dag",
    description="Retail ETL: ingest -> clean -> load warehouse -> RFM features",
    default_args=default_args,
    schedule="@monthly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "etl"],
) as dag:

    download_and_split = PythonOperator(
        task_id="download_and_split",
        python_callable=_download_and_split,
    )

    incremental_ingest = PythonOperator(
        task_id="incremental_ingest",
        python_callable=_incremental_ingest,
    )

    clean_transform = PythonOperator(
        task_id="clean_transform",
        python_callable=_clean,
    )

    load_warehouse = PythonOperator(
        task_id="load_warehouse",
        python_callable=_load,
    )

    rfm_features = PythonOperator(
        task_id="rfm_features",
        python_callable=_features,
    )

    download_and_split >> incremental_ingest >> clean_transform >> load_warehouse >> rfm_features
