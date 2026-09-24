"""Single source of truth for the churn definition, chronological splits and feature list."""
from datetime import date

CHURN_WINDOW_DAYS = 90

TRAIN_CUTOFFS = [date(2011, 3, 1), date(2011, 4, 1)]
VALIDATION_CUTOFFS = [date(2011, 6, 9)]
TEST_CUTOFFS = [date(2011, 9, 9)]

FEATURE_COLUMNS = [
    "recency_days", "frequency", "monetary", "tenure_days", "avg_order_value",
    "avg_days_between_orders", "orders_30d", "orders_60d", "orders_90d",
    "revenue_30d", "revenue_60d", "revenue_90d", "distinct_products", "is_uk",
]
LABEL_COLUMN = "churned"

MODEL_NAME = "churn-model"
MODEL_ALIAS = "champion"
EXPERIMENT_NAME = "churn-prediction"
