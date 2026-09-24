"""Part 2 ML pipeline: features -> monitoring -> (retrain | skip) -> batch scoring.

Retraining only happens when monitoring says the champion needs it AND there are
labelled cutoffs newer than the champion's training data (see ml/monitoring/retraining.py).
"""
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

sys.path.insert(0, "/opt/airflow/project")

from ml.features.build_features import run as build_features  # noqa: E402
from ml.monitoring.drift_monitor import run_monitoring  # noqa: E402
from ml.monitoring.retraining import retrain  # noqa: E402
from ml.scoring.score_customers import run as score_customers  # noqa: E402


def _monitor(**context):
    result = run_monitoring()
    context["ti"].xcom_push(key="retrain", value=result["retrain"])
    context["ti"].xcom_push(key="reasons", value=result["reasons"])


def _branch(**context):
    needs = context["ti"].xcom_pull(key="retrain", task_ids="monitor")
    return "retrain_model" if needs else "skip_retraining"


with DAG(
    dag_id="ml_pipeline_dag",
    description="Churn model: build features, monitor drift/performance, retrain when justified",
    default_args={"owner": "student", "retries": 1},
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "mlops"],
) as dag:
    features = PythonOperator(task_id="build_features", python_callable=build_features)
    monitor = PythonOperator(task_id="monitor", python_callable=_monitor)
    branch = BranchPythonOperator(task_id="decide_retraining", python_callable=_branch)
    retrain_model = PythonOperator(task_id="retrain_model", python_callable=retrain)
    skip = EmptyOperator(task_id="skip_retraining")
    score = PythonOperator(task_id="score_customers", python_callable=score_customers,
                           trigger_rule="none_failed_min_one_success")

    features >> monitor >> branch >> [retrain_model, skip] >> score
