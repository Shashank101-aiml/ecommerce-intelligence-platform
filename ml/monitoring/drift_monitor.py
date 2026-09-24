"""Model monitoring: data quality, feature drift (PSI), model performance and serving health.

Every check is written to warehouse.monitoring_metrics so the dashboard and the
retraining decision read from one place.
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from ml.config import (ERROR_RATE_THRESHOLD, FEATURE_COLUMNS, LABEL_COLUMN, LATENCY_P95_THRESHOLD_MS,
                       MIN_SERVING_SAMPLES, MONITOR_WINDOW_DAYS, NULL_RATE_THRESHOLD, PSI_THRESHOLD,
                       RECALL_DROP_THRESHOLD, TRAIN_CUTOFFS)
from src.db.connection import get_engine
from src.db.upsert import bulk_insert, read_sql_df
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
EPS = 1e-4


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference`."""
    ref = pd.Series(reference, dtype=float).dropna()
    cur = pd.Series(current, dtype=float).dropna()
    if ref.empty or cur.empty:
        return 0.0
    if ref.nunique() <= bins:  # discrete feature such as is_uk: compare category shares
        cats = sorted(set(ref.unique()) | set(cur.unique()))
        p = np.array([(ref == c).mean() for c in cats])
        q = np.array([(cur == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def metric(name, value, threshold=None, feature=None, breached=None) -> dict:
    if breached is None:
        breached = threshold is not None and value > threshold
    return {"metric_name": name, "feature_name": feature, "metric_value": float(value),
            "threshold": threshold, "breached": bool(breached)}


def data_quality_metrics(df: pd.DataFrame) -> list[dict]:
    out = []
    for col in FEATURE_COLUMNS:
        out.append(metric("null_rate", df[col].isna().mean(), NULL_RATE_THRESHOLD, col))
    bad = ((df[FEATURE_COLUMNS] < 0).any(axis=1) | ~df["is_uk"].isin([0, 1])).mean()
    out.append(metric("out_of_range_rate", bad, 0.0))
    return out


def drift_metrics(reference: pd.DataFrame, current: pd.DataFrame) -> list[dict]:
    return [metric("psi", psi(reference[c], current[c]), PSI_THRESHOLD, c) for c in FEATURE_COLUMNS]


def performance_metrics(model, labelled: pd.DataFrame, baseline_recall: float) -> list[dict]:
    X, y = labelled[FEATURE_COLUMNS].astype(float), labelled[LABEL_COLUMN].astype(int)
    pred = model.predict(X)
    recall = recall_score(y, pred, zero_division=0)
    return [
        metric("precision", precision_score(y, pred, zero_division=0), breached=False),
        metric("recall", recall, breached=False),
        metric("recall_drop", baseline_recall - recall, RECALL_DROP_THRESHOLD),
    ]


def serving_metrics(engine, window_days: int = MONITOR_WINDOW_DAYS) -> list[dict]:
    log = read_sql_df(engine, "SELECT status, latency_ms FROM warehouse.prediction_log "
                              "WHERE requested_at >= now() - (%s || ' days')::interval", (str(window_days),))
    if log.empty:
        return []
    lat = log["latency_ms"].dropna().astype(float)
    out = [metric("error_rate", (log["status"] != "ok").mean(), ERROR_RATE_THRESHOLD)]
    if not lat.empty:
        out += [metric("latency_p50_ms", lat.quantile(0.5), breached=False),
                metric("latency_p95_ms", lat.quantile(0.95), LATENCY_P95_THRESHOLD_MS)]
    return out


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    return df


def load_reference(engine, train_cutoffs=None) -> pd.DataFrame:
    """Features of the cutoffs the champion model was trained on."""
    cutoffs = train_cutoffs or TRAIN_CUTOFFS
    dates = ", ".join(f"'{c}'" for c in cutoffs)
    return _features(read_sql_df(engine, f"SELECT * FROM warehouse.ml_customer_features WHERE cutoff_date IN ({dates})"))


def load_current(engine) -> tuple[pd.DataFrame, str]:
    """Recent served features if there are enough, else the latest labelled snapshot."""
    served = read_sql_df(engine, "SELECT features FROM warehouse.prediction_log WHERE status = 'ok' AND features IS NOT NULL "
                                 "AND requested_at >= now() - (%s || ' days')::interval", (str(MONITOR_WINDOW_DAYS),))
    if len(served) >= MIN_SERVING_SAMPLES:
        return _features(pd.DataFrame(list(served["features"]))), "serving_log"
    latest = read_sql_df(engine, "SELECT * FROM warehouse.ml_customer_features WHERE cutoff_date = "
                                 "(SELECT MAX(cutoff_date) FROM warehouse.ml_customer_features WHERE churned IS NOT NULL)")
    return _features(latest), "latest_snapshot"


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Demo/testing aid: customers suddenly buy far less recently."""
    d = df.copy()
    d["recency_days"] *= 3
    for col in ("orders_30d", "orders_60d", "orders_90d", "revenue_30d", "revenue_60d", "revenue_90d"):
        d[col] *= 0.2
    return d


def save_metrics(engine, metrics: list[dict]) -> None:
    bulk_insert(engine, "warehouse", "monitoring_metrics", pd.DataFrame(metrics))


def run_monitoring(simulate: bool = False) -> dict:
    from ml.monitoring.retraining import champion_params, decide_retraining, load_champion_and_baseline
    engine = get_engine()
    trained_on = re.findall(r"\d{4}-\d{2}-\d{2}", champion_params().get("train_cutoffs", ""))
    reference = load_reference(engine, trained_on or None)
    current, source = load_current(engine)
    if simulate:
        current, source = simulate_drift(current), source + "+simulated_drift"

    model, baseline_recall = load_champion_and_baseline()
    labelled = current if LABEL_COLUMN in current and current[LABEL_COLUMN].notna().all() else None
    metrics = data_quality_metrics(current) + drift_metrics(reference, current) + serving_metrics(engine)
    if labelled is not None:
        metrics += performance_metrics(model, labelled, baseline_recall)
    save_metrics(engine, metrics)

    retrain, reasons = decide_retraining(metrics)
    logger.info("monitoring source=%s breaches=%d retrain=%s reasons=%s", source,
                sum(m["breached"] for m in metrics), retrain, reasons)
    return {"source": source, "metrics": metrics, "retrain": retrain, "reasons": reasons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-drift", action="store_true")
    result = run_monitoring(parser.parse_args().simulate_drift)
    print({"source": result["source"], "retrain": result["retrain"], "reasons": result["reasons"]})
