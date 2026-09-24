"""Churn prediction service: serves the MLflow-registered `champion` model."""
import json
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from api.schemas import CustomerPredictResponse, HealthResponse, PredictRequest, PredictResponse
from ml.config import FEATURE_COLUMNS, MODEL_ALIAS, MODEL_NAME
from ml.features.build_features import build_features
from ml.mlflow_tracking.mlflow_config import DEFAULT_TRACKING_URI
from src.db.connection import get_engine
from src.db.upsert import read_sql_df
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
THRESHOLD = 0.5

CUSTOMER_ORDERS_QUERY = """
    SELECT f.customer_key, d.full_date, f.invoice_no, f.product_key, f.revenue, g.country
    FROM warehouse.fact_sales f
    JOIN warehouse.dim_date d ON f.date_key = d.date_key
    JOIN warehouse.dim_geography g ON f.geography_key = g.geography_key
    JOIN warehouse.dim_customer c ON f.customer_key = c.customer_key
    WHERE c.customer_id = %s
"""


def load_champion():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return model, str(version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model, app.state.model_version = load_champion()
        logger.info("loaded %s@%s (version %s)", MODEL_NAME, MODEL_ALIAS, app.state.model_version)
    except Exception as exc:  # noqa: BLE001 - service stays up and reports unhealthy
        logger.error("could not load model: %s", exc)
        app.state.model, app.state.model_version = None, None
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)
app.state.model = None
app.state.model_version = None


def log_prediction(*, customer_key, model_version, features, probability, label, latency_ms, status, error=None):
    """Best-effort audit log; logging problems must never break a prediction."""
    try:
        conn = get_engine().raw_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO warehouse.prediction_log
                   (customer_key, model_version, features, churn_probability, predicted_label, latency_ms, status, error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (customer_key, model_version, json.dumps(features) if features is not None else None,
                 probability, label, latency_ms, status, error))
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("prediction_log write failed: %s", exc)


def _score(features: dict) -> tuple[float, int]:
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS).astype(float)
    probability = float(app.state.model.predict_proba(frame)[0][1])
    return probability, int(probability >= THRESHOLD)


def _require_model():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = app.state.model is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          model_version=app.state.model_version)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _require_model()
    features = request.model_dump()
    start = time.perf_counter()
    try:
        probability, label = _score(features)
    except Exception as exc:  # noqa: BLE001
        log_prediction(customer_key=None, model_version=app.state.model_version, features=features,
                       probability=None, label=None, latency_ms=(time.perf_counter() - start) * 1000,
                       status="error", error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    log_prediction(customer_key=None, model_version=app.state.model_version, features=features,
                   probability=probability, label=label, latency_ms=(time.perf_counter() - start) * 1000,
                   status="ok")
    return PredictResponse(churn_probability=probability, predicted_label=label,
                           model_version=app.state.model_version)


@app.post("/predict/customer/{customer_id}", response_model=CustomerPredictResponse)
def predict_customer(customer_id: str):
    _require_model()
    start = time.perf_counter()
    engine = get_engine()
    orders = read_sql_df(engine, CUSTOMER_ORDERS_QUERY, (customer_id,))
    if orders.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    orders["full_date"] = pd.to_datetime(orders["full_date"])
    orders["revenue"] = orders["revenue"].astype(float)

    as_of = pd.to_datetime(read_sql_df(engine, "SELECT MAX(full_date) AS d FROM warehouse.dim_date")["d"].iloc[0])
    feats = build_features(orders, as_of.date(), with_label=False)
    features = {c: float(feats.iloc[0][c]) for c in FEATURE_COLUMNS}
    probability, label = _score(features)
    log_prediction(customer_key=int(feats.iloc[0]["customer_key"]), model_version=app.state.model_version,
                   features=features, probability=probability, label=label,
                   latency_ms=(time.perf_counter() - start) * 1000, status="ok")
    return CustomerPredictResponse(customer_id=customer_id, as_of_date=str(as_of.date()),
                                   churn_probability=probability, predicted_label=label,
                                   model_version=app.state.model_version)
