"""Trains LogReg / RandomForest / XGBoost churn models, tracks every run in MLflow,
selects the best on the validation split, evaluates it once on the test split and
registers it in the MLflow Model Registry with the `champion` alias.
"""
import itertools
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.config import (FEATURE_COLUMNS, LABEL_COLUMN, MODEL_ALIAS, MODEL_NAME, SKOPS_TRUSTED_TYPES,
                       TEST_CUTOFFS, TRAIN_CUTOFFS, VALIDATION_CUTOFFS)
from ml.mlflow_tracking.mlflow_config import configure_mlflow
from src.db.connection import get_engine
from src.db.upsert import read_sql_df
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
SEED = 42


def load_split(engine, cutoffs) -> pd.DataFrame:
    dates = ", ".join(f"'{c}'" for c in cutoffs)
    df = read_sql_df(engine, f"SELECT * FROM warehouse.ml_customer_features WHERE cutoff_date IN ({dates})")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    return df


def model_grid():
    for c in (0.1, 1.0, 10.0):
        yield "logistic_regression", {"C": c}, lambda p: make_pipeline(
            StandardScaler(), LogisticRegression(C=p["C"], class_weight="balanced", max_iter=1000, random_state=SEED))
    for n, depth in itertools.product((100, 300), (6, 12)):
        yield "random_forest", {"n_estimators": n, "max_depth": depth}, lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], class_weight="balanced", n_jobs=-1, random_state=SEED)
    for n, lr, depth in itertools.product((100, 300), (0.05, 0.1), (3, 5)):
        yield "xgboost", {"n_estimators": n, "learning_rate": lr, "max_depth": depth}, lambda p: XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            eval_metric="logloss", random_state=SEED)


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def log_artifacts(model, X, y, prefix: str):
    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = confusion_matrix(y, model.predict(X))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"{prefix} confusion matrix")
        path = Path(tmp) / f"{prefix}_confusion_matrix.png"
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        mlflow.log_artifact(str(path))

        est = model[-1] if hasattr(model, "steps") else model
        importance = getattr(est, "feature_importances_", None)
        if importance is None and hasattr(est, "coef_"):
            importance = np.abs(est.coef_[0])
        if importance is not None:
            imp = pd.Series(importance, index=FEATURE_COLUMNS).sort_values(ascending=False)
            csv = Path(tmp) / "feature_importance.csv"
            imp.to_csv(csv, header=["importance"])
            mlflow.log_artifact(str(csv))


def run(train_cutoffs=None, validation_cutoffs=None, test_cutoffs=None) -> dict:
    train_cutoffs = train_cutoffs or TRAIN_CUTOFFS
    validation_cutoffs = validation_cutoffs or VALIDATION_CUTOFFS
    test_cutoffs = test_cutoffs or TEST_CUTOFFS
    uri = configure_mlflow()
    engine = get_engine()
    train, val, test = (load_split(engine, c) for c in (train_cutoffs, validation_cutoffs, test_cutoffs))
    X_tr, y_tr = train[FEATURE_COLUMNS], train[LABEL_COLUMN].astype(int)
    X_va, y_va = val[FEATURE_COLUMNS], val[LABEL_COLUMN].astype(int)
    X_te, y_te = test[FEATURE_COLUMNS], test[LABEL_COLUMN].astype(int)
    logger.info("rows train=%d val=%d test=%d (tracking: %s)", len(train), len(val), len(test), uri)

    best = None
    with mlflow.start_run(run_name="model-selection"):
        iso = lambda cs: ",".join(str(c) for c in cs)  # noqa: E731
        mlflow.log_params({"train_cutoffs": iso(train_cutoffs), "validation_cutoffs": iso(validation_cutoffs),
                           "test_cutoffs": iso(test_cutoffs),
                           "data_max_cutoff": str(max([*train_cutoffs, *validation_cutoffs, *test_cutoffs]))})
        for family, params, factory in model_grid():
            with mlflow.start_run(run_name=f"{family}", nested=True):
                model = factory(params).fit(X_tr, y_tr)
                val_metrics = evaluate(model, X_va, y_va)
                mlflow.log_params({"model_family": family, **params})
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                mlflow.log_metrics({f"train_{k}": v for k, v in evaluate(model, X_tr, y_tr).items()})
                log_artifacts(model, X_va, y_va, "val")
                if best is None or val_metrics["pr_auc"] > best["val"]["pr_auc"]:
                    best = {"family": family, "params": params, "model": model, "val": val_metrics}

        # Test split is touched exactly once, for the selected model only.
        test_metrics = evaluate(best["model"], X_te, y_te)
        mlflow.log_params({"best_family": best["family"], **{f"best_{k}": v for k, v in best["params"].items()}})
        mlflow.log_metrics({f"val_{k}": v for k, v in best["val"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log_artifacts(best["model"], X_te, y_te, "test")
        info = mlflow.sklearn.log_model(
            best["model"], artifact_path="model",
            signature=infer_signature(X_te, best["model"].predict_proba(X_te)[:, 1]),
            registered_model_name=MODEL_NAME, skops_trusted_types=SKOPS_TRUSTED_TYPES)

    client = MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    summary = {"best_family": best["family"], "params": best["params"], "registered_version": version,
               "val": best["val"], "test": test_metrics}
    logger.info("training summary: %s", summary)
    return summary


if __name__ == "__main__":
    run()
