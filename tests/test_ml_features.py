from datetime import date

import pandas as pd
import pytest

from ml.features.build_features import build_features


def _orders(rows):
    df = pd.DataFrame(rows, columns=["customer_key", "full_date", "invoice_no", "product_key", "revenue", "country"])
    df["full_date"] = pd.to_datetime(df["full_date"])
    return df


ORDERS = _orders([
    (1, "2011-01-05", "A1", 10, 20.0, "United Kingdom"),
    (1, "2011-01-25", "A2", 11, 30.0, "United Kingdom"),
    (1, "2011-03-15", "A3", 10, 50.0, "United Kingdom"),   # after cutoff, inside window
    (2, "2011-01-10", "B1", 12, 40.0, "France"),            # never returns -> churn
    (3, "2011-02-20", "C1", 10, 15.0, "United Kingdom"),    # first purchase after cutoff
    (4, "2011-06-30", "D1", 10, 15.0, "United Kingdom"),    # pads data range
])
CUTOFF = date(2011, 2, 1)


def test_features_ignore_data_after_cutoff():
    feats = build_features(ORDERS, CUTOFF).set_index("customer_key")
    assert set(feats.index) == {1, 2}
    assert feats.loc[1, "frequency"] == 2
    assert feats.loc[1, "monetary"] == 50.0
    assert feats.loc[1, "recency_days"] == 7


def test_future_purchase_does_not_change_features():
    later = _orders([(1, "2011-03-16", "A4", 13, 999.0, "United Kingdom")])
    extended = pd.concat([ORDERS, later], ignore_index=True)
    a = build_features(ORDERS, CUTOFF).set_index("customer_key").loc[1]
    b = build_features(extended, CUTOFF).set_index("customer_key").loc[1]
    assert a.drop("churned").equals(b.drop("churned"))


def test_churn_label_uses_window_after_cutoff():
    feats = build_features(ORDERS, CUTOFF, window_days=90).set_index("customer_key")
    assert feats.loc[1, "churned"] == 0
    assert feats.loc[2, "churned"] == 1


def test_label_window_beyond_data_raises():
    with pytest.raises(ValueError):
        build_features(ORDERS, date(2011, 6, 1), window_days=90)


def test_unlabeled_features_for_scoring():
    feats = build_features(ORDERS, CUTOFF, with_label=False)
    assert feats["churned"].isna().all()
