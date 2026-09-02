-- Analytical marts consumed directly by the Streamlit dashboard.
CREATE SCHEMA IF NOT EXISTS marts;

-- DROP + CREATE (not CREATE OR REPLACE) because Postgres refuses to
-- reorder/insert columns in an existing view definition.
DROP VIEW IF EXISTS marts.sales_mart;
CREATE VIEW marts.sales_mart AS
SELECT
    f.sales_key,
    f.invoice_no,
    d.full_date,
    d.year,
    d.month,
    d.month_name,
    p.stock_code,
    p.description,
    g.country,
    f.customer_key,
    f.quantity,
    f.unit_price,
    f.revenue
FROM warehouse.fact_sales f
JOIN warehouse.dim_date d ON f.date_key = d.date_key
JOIN warehouse.dim_product p ON f.product_key = p.product_key
JOIN warehouse.dim_geography g ON f.geography_key = g.geography_key;

DROP VIEW IF EXISTS marts.customer_mart;
CREATE VIEW marts.customer_mart AS
SELECT
    c.customer_key,
    c.customer_id,
    c.country,
    c.first_purchase_date,
    r.snapshot_date,
    r.recency_days,
    r.frequency,
    r.monetary,
    r.rfm_segment
FROM warehouse.dim_customer c
LEFT JOIN warehouse.fact_customer_retention r ON c.customer_key = r.customer_key;
