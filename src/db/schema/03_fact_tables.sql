-- Fact tables. fact_sales is deduplicated on (invoice_no, stock_code) so
-- reloading the same source rows is idempotent (supports incremental loads).
CREATE TABLE IF NOT EXISTS warehouse.fact_sales (
    sales_key      SERIAL PRIMARY KEY,
    invoice_no     TEXT NOT NULL,
    stock_code     TEXT NOT NULL,
    date_key       INTEGER NOT NULL REFERENCES warehouse.dim_date(date_key),
    customer_key   INTEGER REFERENCES warehouse.dim_customer(customer_key),
    product_key    INTEGER NOT NULL REFERENCES warehouse.dim_product(product_key),
    geography_key  INTEGER NOT NULL REFERENCES warehouse.dim_geography(geography_key),
    quantity       INTEGER NOT NULL,
    unit_price     NUMERIC(12, 2) NOT NULL,
    revenue        NUMERIC(14, 2) NOT NULL,
    loaded_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (invoice_no, stock_code)
);

CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON warehouse.fact_sales(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_sales_customer ON warehouse.fact_sales(customer_key);

CREATE TABLE IF NOT EXISTS warehouse.fact_customer_retention (
    retention_key   SERIAL PRIMARY KEY,
    customer_key    INTEGER NOT NULL REFERENCES warehouse.dim_customer(customer_key),
    snapshot_date   DATE NOT NULL,
    recency_days    INTEGER NOT NULL,
    frequency       INTEGER NOT NULL,
    monetary        NUMERIC(14, 2) NOT NULL,
    rfm_segment     TEXT NOT NULL,
    UNIQUE (customer_key, snapshot_date)
);
