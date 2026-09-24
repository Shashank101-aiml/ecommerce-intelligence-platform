-- Part 2: point-in-time ML features, prediction log and monitoring metrics.
CREATE TABLE IF NOT EXISTS warehouse.ml_customer_features (
    customer_key             INTEGER NOT NULL REFERENCES warehouse.dim_customer(customer_key),
    cutoff_date              DATE NOT NULL,
    recency_days             INTEGER NOT NULL,
    frequency                INTEGER NOT NULL,
    monetary                 NUMERIC(14, 2) NOT NULL,
    tenure_days              INTEGER NOT NULL,
    avg_order_value          NUMERIC(14, 2) NOT NULL,
    avg_days_between_orders  NUMERIC(10, 2) NOT NULL,
    orders_30d               INTEGER NOT NULL,
    orders_60d               INTEGER NOT NULL,
    orders_90d               INTEGER NOT NULL,
    revenue_30d              NUMERIC(14, 2) NOT NULL,
    revenue_60d              NUMERIC(14, 2) NOT NULL,
    revenue_90d              NUMERIC(14, 2) NOT NULL,
    distinct_products        INTEGER NOT NULL,
    is_uk                    SMALLINT NOT NULL,
    churned                  SMALLINT,
    PRIMARY KEY (customer_key, cutoff_date)
);

CREATE TABLE IF NOT EXISTS warehouse.prediction_log (
    id               SERIAL PRIMARY KEY,
    requested_at     TIMESTAMP NOT NULL DEFAULT now(),
    customer_key     INTEGER,
    model_version    TEXT,
    features         JSONB,
    churn_probability DOUBLE PRECISION,
    predicted_label  SMALLINT,
    latency_ms       DOUBLE PRECISION,
    status           TEXT NOT NULL,
    error            TEXT
);

CREATE TABLE IF NOT EXISTS warehouse.monitoring_metrics (
    id             SERIAL PRIMARY KEY,
    measured_at    TIMESTAMP NOT NULL DEFAULT now(),
    metric_name    TEXT NOT NULL,
    feature_name   TEXT,
    metric_value   DOUBLE PRECISION,
    threshold      DOUBLE PRECISION,
    breached       BOOLEAN NOT NULL DEFAULT FALSE
);

-- Batch churn scores (one row per customer per scoring date), read by the dashboard.
CREATE TABLE IF NOT EXISTS warehouse.customer_churn_scores (
    customer_key       INTEGER NOT NULL REFERENCES warehouse.dim_customer(customer_key),
    as_of_date         DATE NOT NULL,
    churn_probability  DOUBLE PRECISION NOT NULL,
    predicted_label    SMALLINT NOT NULL,
    risk_tier          TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    scored_at          TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_key, as_of_date)
);
