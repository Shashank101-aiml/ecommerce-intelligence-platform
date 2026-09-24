"""Batch-scores every customer with the champion model as of the latest data date."""
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from ml.config import (FEATURE_COLUMNS, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, MODEL_ALIAS, MODEL_NAME)
from ml.features.build_features import build_features, load_orders
from ml.mlflow_tracking.mlflow_config import configure_mlflow
from src.db.connection import get_engine
from src.db.upsert import read_sql_df, upsert
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def risk_tier(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def run() -> dict:
    configure_mlflow()
    version = str(MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    engine = get_engine()
    orders = load_orders(engine)
    as_of = pd.to_datetime(read_sql_df(engine, "SELECT MAX(full_date) AS d FROM warehouse.dim_date")["d"].iloc[0]).date()
    feats = build_features(orders, as_of, with_label=False)

    proba = model.predict_proba(feats[FEATURE_COLUMNS].astype(float))[:, 1]
    scores = pd.DataFrame({
        "customer_key": feats["customer_key"], "as_of_date": as_of,
        "churn_probability": proba, "predicted_label": (proba >= 0.5).astype(int),
        "risk_tier": [risk_tier(p) for p in proba], "model_version": version,
    })
    upsert(engine, "warehouse", "customer_churn_scores", scores, conflict_cols=["customer_key", "as_of_date"],
           update_cols=["churn_probability", "predicted_label", "risk_tier", "model_version"])
    summary = {"as_of": str(as_of), "model_version": version, "customers": len(scores),
               **scores["risk_tier"].value_counts().to_dict()}
    logger.info("scoring summary: %s", summary)
    return summary


if __name__ == "__main__":
    run()
