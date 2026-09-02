-- Star schema dimensions.
CREATE SCHEMA IF NOT EXISTS warehouse;

CREATE TABLE IF NOT EXISTS warehouse.dim_date (
    date_key      INTEGER PRIMARY KEY,      -- YYYYMMDD
    full_date     DATE NOT NULL UNIQUE,
    day           SMALLINT NOT NULL,
    month         SMALLINT NOT NULL,
    month_name    TEXT NOT NULL,
    quarter       SMALLINT NOT NULL,
    year          SMALLINT NOT NULL,
    day_of_week   TEXT NOT NULL,
    is_weekend    BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS warehouse.dim_customer (
    customer_key         SERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL UNIQUE,
    country              TEXT,
    first_purchase_date  DATE,
    created_at           TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS warehouse.dim_product (
    product_key   SERIAL PRIMARY KEY,
    stock_code    TEXT NOT NULL UNIQUE,
    description   TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS warehouse.dim_geography (
    geography_key SERIAL PRIMARY KEY,
    country       TEXT NOT NULL UNIQUE
);
