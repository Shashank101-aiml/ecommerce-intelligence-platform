import numpy as np
import pandas as pd

from ml.config import FEATURE_COLUMNS
from ml.monitoring.drift_monitor import (data_quality_metrics, drift_metrics, metric, psi, simulate_drift)
from ml.monitoring.retraining import decide_retraining


def _frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.gamma(2.0, 10.0, n) for c in FEATURE_COLUMNS})
    df["is_uk"] = rng.integers(0, 2, n)
    return df


def test_psi_is_zero_for_identical_and_large_for_shifted():
    x = np.random.default_rng(1).normal(size=2000)
    assert psi(x, x) < 1e-6
    assert psi(x, x + 3) > 0.5


def test_psi_handles_discrete_and_constant_features():
    ref = pd.Series([0, 1] * 100)
    assert psi(ref, pd.Series([0] * 200)) > PSI_LIMIT
    assert psi(pd.Series([5.0] * 50), pd.Series([5.0] * 50)) == 0.0


PSI_LIMIT = 0.2


def test_no_drift_between_samples_from_same_distribution():
    m = drift_metrics(_frame(seed=1), _frame(seed=2))
    assert not any(x["breached"] for x in m)


def test_simulated_drift_breaches_multiple_features():
    ref = _frame()
    breached = [x for x in drift_metrics(ref, simulate_drift(ref)) if x["breached"]]
    assert len(breached) >= 2


def test_data_quality_flags_nulls_and_out_of_range():
    df = _frame(100)
    df.loc[:19, "frequency"] = None
    df.loc[0, "monetary"] = -5
    by_name = {(m["metric_name"], m["feature_name"]): m for m in data_quality_metrics(df)}
    assert by_name[("null_rate", "frequency")]["breached"]
    assert by_name[("out_of_range_rate", None)]["breached"]


def test_retraining_decision_rules():
    drift = [metric("psi", 0.5, 0.2, f) for f in ("recency_days", "frequency")]
    assert decide_retraining(drift)[0] is True
    assert decide_retraining(drift[:1])[0] is False              # one drifted feature is not enough
    assert decide_retraining([metric("recall_drop", 0.2, 0.1)])[0] is True
    assert decide_retraining([metric("error_rate", 0.05, 0.02)])[0] is True
    assert decide_retraining([metric("recall_drop", 0.02, 0.1), metric("error_rate", 0.0, 0.02)]) == (False, [])


def test_rolling_windows_are_chronological_and_disjoint():
    from datetime import date
    from ml.monitoring.retraining import rolling_windows
    cutoffs = [date(2011, 9, 9), date(2011, 3, 1), date(2011, 6, 9), date(2011, 4, 1)]
    train, val, test = rolling_windows(cutoffs)
    assert train == [date(2011, 3, 1), date(2011, 4, 1)] and val == [date(2011, 6, 9)] and test == [date(2011, 9, 9)]
    assert max(train) < min(val) < min(test)


def test_rolling_windows_need_three_cutoffs():
    import pytest
    from datetime import date
    from ml.monitoring.retraining import rolling_windows
    with pytest.raises(ValueError):
        rolling_windows([date(2011, 3, 1), date(2011, 4, 1)])


def test_retrain_is_deferred_without_new_labelled_data(monkeypatch):
    from datetime import date
    import ml.monitoring.retraining as r
    monkeypatch.setattr(r, "get_engine", lambda: None)
    monkeypatch.setattr(r, "labelled_cutoffs", lambda e: [date(2011, 3, 1), date(2011, 9, 9)])
    monkeypatch.setattr(r, "champion_params", lambda: {"data_max_cutoff": "2011-09-09"})
    assert r.retrain()["retrained"] is False
    monkeypatch.setattr(r, "champion_params", lambda: {"data_max_cutoff": "2011-06-09"})
    assert r.has_new_training_data(None) is True
