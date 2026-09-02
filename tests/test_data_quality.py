import pandas as pd

from src.etl.data_quality_checks import validate_dataframe


def _base_row(**overrides):
    row = {
        "invoice_no": "536365",
        "stock_code": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "quantity": 6,
        "invoice_date": pd.Timestamp("2010-12-01 08:26:00"),
        "unit_price": 2.55,
        "customer_id": "17850",
        "country": "United Kingdom",
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    df = pd.DataFrame([_base_row()])
    valid, rejected = validate_dataframe(df)
    assert len(valid) == 1
    assert rejected.empty


def test_cancelled_order_is_rejected():
    df = pd.DataFrame([_base_row(invoice_no="C536365")])
    valid, rejected = validate_dataframe(df)
    assert valid.empty
    assert rejected.iloc[0]["rejection_reason"] == "cancelled order"


def test_non_positive_quantity_is_rejected():
    df = pd.DataFrame([_base_row(quantity=-3)])
    valid, rejected = validate_dataframe(df)
    assert valid.empty
    assert rejected.iloc[0]["rejection_reason"] == "non-positive quantity"


def test_non_positive_price_is_rejected():
    df = pd.DataFrame([_base_row(unit_price=0)])
    valid, rejected = validate_dataframe(df)
    assert valid.empty
    assert rejected.iloc[0]["rejection_reason"] == "non-positive unit_price"


def test_missing_country_is_rejected():
    df = pd.DataFrame([_base_row(country=None)])
    valid, rejected = validate_dataframe(df)
    assert valid.empty
    assert rejected.iloc[0]["rejection_reason"] == "missing country"


def test_mixed_batch_splits_correctly():
    df = pd.DataFrame([_base_row(), _base_row(invoice_no="C999999")])
    valid, rejected = validate_dataframe(df)
    assert len(valid) == 1
    assert len(rejected) == 1
