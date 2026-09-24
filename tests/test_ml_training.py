import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.config import FEATURE_COLUMNS
from ml.training.train_model import evaluate, model_grid


def test_grid_covers_all_three_model_families():
    families = {family for family, _, _ in model_grid()}
    assert families == {"logistic_regression", "random_forest", "xgboost"}


def test_every_grid_entry_builds_a_fittable_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X.iloc[:, 0] > 0).astype(int)
    seen = set()
    for family, params, factory in model_grid():
        if family in seen:
            continue
        seen.add(family)
        assert hasattr(factory(params).fit(X, y), "predict_proba")


def test_evaluate_returns_all_metrics_in_range():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = (X["a"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    metrics = evaluate(LogisticRegression().fit(X, y), X, y)
    assert set(metrics) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["roc_auc"] > 0.8
