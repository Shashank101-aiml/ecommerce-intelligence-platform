-- Raw/staging layer: holds ingested rows close to their source shape,
-- plus a table for rows that fail data-quality validation.
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.sales_raw (
    id            SERIAL PRIMARY KEY,
    invoice_no    TEXT,
    stock_code    TEXT,
    description   TEXT,
    quantity      INTEGER,
    invoice_date  TIMESTAMP,
    unit_price    NUMERIC(12, 2),
    customer_id   TEXT,
    country       TEXT,
    source_file   TEXT,
    batch_id      TEXT,
    loaded_at     TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sales_raw_batch ON staging.sales_raw(batch_id);

CREATE TABLE IF NOT EXISTS staging.rejected_records (
    id            SERIAL PRIMARY KEY,
    raw_data      JSONB,
    reason        TEXT NOT NULL,
    source_file   TEXT,
    batch_id      TEXT,
    rejected_at   TIMESTAMP NOT NULL DEFAULT now()
);
