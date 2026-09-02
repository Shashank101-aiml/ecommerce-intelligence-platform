import pandas as pd

from src.etl.clean_transform import _standardize


def test_standardize_renames_and_types_columns():
    raw = pd.DataFrame([{
        "InvoiceNo": " 536365 ",
        "StockCode": " 85123A ",
        "Description": " WHITE HANGING HEART T-LIGHT HOLDER ",
        "Quantity": "6",
        "InvoiceDate": "2010-12-01 08:26:00",
        "UnitPrice": "2.5495",
        "CustomerID": 17850.0,
        "Country": " United Kingdom ",
    }])

    out = _standardize(raw)

    assert out.loc[0, "invoice_no"] == "536365"
    assert out.loc[0, "stock_code"] == "85123A"
    assert out.loc[0, "customer_id"] == "17850"
    assert out.loc[0, "unit_price"] == 2.55
    assert out.loc[0, "quantity"] == 6
    assert pd.notna(out.loc[0, "invoice_date"])


def test_standardize_handles_missing_customer_id():
    raw = pd.DataFrame([{
        "InvoiceNo": "536365", "StockCode": "85123A", "Description": "ITEM",
        "Quantity": "6", "InvoiceDate": "2010-12-01", "UnitPrice": "2.55",
        "CustomerID": None, "Country": "United Kingdom",
    }])

    out = _standardize(raw)

    assert out.loc[0, "customer_id"] is None
