"""Validation rules applied to standardized sales rows.

Called by clean_transform before anything reaches the staging CSV / DB,
so the final analytical dataset is never hand-edited and every rejection
carries a reason (see docs/data_dictionary.md for the full rule list).
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "invoice_no", "stock_code", "description", "quantity",
    "invoice_date", "unit_price", "customer_id", "country",
]


def _first_matching_reason(row: pd.Series) -> str | None:
    if not row["invoice_no"] or pd.isna(row["invoice_no"]):
        return "missing invoice_no"
    if str(row["invoice_no"]).startswith("C"):
        return "cancelled order"
    if not row["stock_code"] or pd.isna(row["stock_code"]):
        return "missing stock_code"
    if pd.isna(row["invoice_date"]):
        return "missing or unparseable invoice_date"
    if pd.isna(row["quantity"]) or row["quantity"] <= 0:
        return "non-positive quantity"
    if pd.isna(row["unit_price"]) or row["unit_price"] <= 0:
        return "non-positive unit_price"
    if not row["country"] or pd.isna(row["country"]):
        return "missing country"
    return None


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits df into (valid_rows, rejected_rows). rejected_rows gains a
    'rejection_reason' column explaining why each row was dropped."""
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input dataframe missing required columns: {missing_cols}")

    reasons = df.apply(_first_matching_reason, axis=1)
    rejected_mask = reasons.notna()

    rejected_df = df[rejected_mask].copy()
    rejected_df["rejection_reason"] = reasons[rejected_mask]

    valid_df = df[~rejected_mask].copy()
    return valid_df, rejected_df
