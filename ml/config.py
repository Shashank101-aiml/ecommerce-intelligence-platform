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

# Types skops must trust when (de)serialising models we trained ourselves.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Monitoring / retraining thresholds
PSI_THRESHOLD = 0.2              # >0.2 is a conventional "significant shift"
PSI_FEATURES_BREACH_MIN = 2      # retrain if at least this many features drift
RECALL_DROP_THRESHOLD = 0.10     # absolute drop vs the champion's recorded test recall
ERROR_RATE_THRESHOLD = 0.02
LATENCY_P95_THRESHOLD_MS = 1000.0
NULL_RATE_THRESHOLD = 0.05
MIN_SERVING_SAMPLES = 30         # below this, drift is measured on the latest labelled snapshot
MONITOR_WINDOW_DAYS = 30

# Risk tiers for the dashboard (probability of churn within CHURN_WINDOW_DAYS)
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
