"""Retraining policy: when monitoring metrics justify training a new champion."""
import mlflow
from mlflow.tracking import MlflowClient

from datetime import date

from ml.config import ERROR_RATE_THRESHOLD, MODEL_ALIAS, MODEL_NAME, PSI_FEATURES_BREACH_MIN, RECALL_DROP_THRESHOLD
from ml.mlflow_tracking.mlflow_config import configure_mlflow
from src.db.connection import get_engine
from src.db.upsert import read_sql_df
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def decide_retraining(metrics: list[dict]) -> tuple[bool, list[str]]:
    """Retrain when (a) enough features drifted, (b) recall fell too far, or (c) serving errors are high."""
    reasons = []
    drifted = [m["feature_name"] for m in metrics if m["metric_name"] == "psi" and m["breached"]]
    if len(drifted) >= PSI_FEATURES_BREACH_MIN:
        reasons.append(f"feature drift: {len(drifted)} features with PSI above threshold ({', '.join(drifted)})")
    for m in metrics:
        if m["metric_name"] == "recall_drop" and m["breached"]:
            reasons.append(f"recall dropped {m['metric_value']:.2f} vs baseline (limit {RECALL_DROP_THRESHOLD})")
        if m["metric_name"] == "error_rate" and m["breached"]:
            reasons.append(f"serving error rate {m['metric_value']:.1%} above {ERROR_RATE_THRESHOLD:.0%}")
    return bool(reasons), reasons


def load_champion_and_baseline():
    """Champion model plus the test recall it was registered with (the performance baseline)."""
    configure_mlflow()
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    baseline = client.get_run(version.run_id).data.metrics.get("test_recall", 0.0)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}"), baseline


def labelled_cutoffs(engine) -> list[date]:
    df = read_sql_df(engine, "SELECT DISTINCT cutoff_date FROM warehouse.ml_customer_features "
                             "WHERE churned IS NOT NULL ORDER BY 1")
    return list(df["cutoff_date"])


def rolling_windows(cutoffs: list[date]) -> tuple[list[date], list[date], list[date]]:
    """Chronological split: newest cutoff = test, the one before = validation, the rest = train."""
    if len(cutoffs) < 3:
        raise ValueError("need at least 3 labelled cutoffs to build train/validation/test windows")
    ordered = sorted(cutoffs)
    return ordered[:-2], [ordered[-2]], [ordered[-1]]


def champion_params() -> dict:
    configure_mlflow()
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    return client.get_run(version.run_id).data.params


def has_new_training_data(engine) -> bool:
    """Retraining on the same labelled cutoffs the champion already saw would just recreate it."""
    seen = champion_params().get("data_max_cutoff")
    if seen is None:  # champion predates this bookkeeping: allow one retrain to stamp it
        return True
    return max(labelled_cutoffs(engine)) > date.fromisoformat(seen)


def retrain(force: bool = False) -> dict:
    """Trains and registers a new model version (which becomes champion) on rolled-forward windows."""
    from ml.training.train_model import run as train
    engine = get_engine()
    if not force and not has_new_training_data(engine):
        logger.info("retraining deferred: no labelled cutoffs newer than the champion's training data")
        return {"retrained": False, "reason": "no new labelled data"}
    train_c, val_c, test_c = rolling_windows(labelled_cutoffs(engine))
    logger.info("retraining on train=%s val=%s test=%s", train_c, val_c, test_c)
    return {"retrained": True, **train(train_c, val_c, test_c)}
