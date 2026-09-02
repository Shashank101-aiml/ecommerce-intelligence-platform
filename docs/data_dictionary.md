# Data Dictionary & Validation Rules

## Source dataset

**UCI Online Retail Dataset** (`https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx`)
Transaction-level UK online retailer data, 01-Dec-2010 to 09-Dec-2011. Free, public, no access restrictions. Kaggle datasets were considered (optional per the assignment) but not used — the incremental-ingestion requirement is satisfied instead by slicing this same dataset into per-month CSV extracts (see `src/ingestion/split_into_monthly_extracts.py`).

| Source column | Description |
|---|---|
| InvoiceNo | Invoice number. Prefix "C" = cancelled order. |
| StockCode | Product/item code. |
| Description | Product name. |
| Quantity | Units per transaction line. |
| InvoiceDate | Date/time of the transaction. |
| UnitPrice | Unit price in GBP. |
| CustomerID | Customer identifier (blank for guest/unidentified purchases). |
| Country | Customer's country. |

## Validation rules (`src/etl/data_quality_checks.py`)

A row is rejected (written to `data/rejected/` and `staging.rejected_records`, never loaded into the warehouse) if any of the following apply, in this order:

| # | Rule | Reason logged |
|---|---|---|
| 1 | `invoice_no` missing | `missing invoice_no` |
| 2 | `invoice_no` starts with `C` | `cancelled order` |
| 3 | `stock_code` missing | `missing stock_code` |
| 4 | `invoice_date` missing/unparseable | `missing or unparseable invoice_date` |
| 5 | `quantity` missing or ≤ 0 | `non-positive quantity` |
| 6 | `unit_price` missing or ≤ 0 | `non-positive unit_price` |
| 7 | `country` missing | `missing country` |

Currency: the source dataset is entirely GBP; `unit_price` is standardized to 2 decimal places. No FX conversion is performed.

## Warehouse schema (`src/db/schema/`)

### `warehouse.dim_date`
| Column | Type | Notes |
|---|---|---|
| date_key | INTEGER PK | `YYYYMMDD` |
| full_date | DATE | unique |
| day, month, quarter, year | SMALLINT | |
| month_name, day_of_week | TEXT | |
| is_weekend | BOOLEAN | |

### `warehouse.dim_customer`
| Column | Type | Notes |
|---|---|---|
| customer_key | SERIAL PK | surrogate key |
| customer_id | TEXT | natural key from source, unique; rows with a blank source `CustomerID` never get a dim_customer row and load into `fact_sales` with `customer_key = NULL` |
| country | TEXT | customer's first-seen country |
| first_purchase_date | DATE | earliest invoice date seen for this customer |

### `warehouse.dim_product`
| Column | Type | Notes |
|---|---|---|
| product_key | SERIAL PK | surrogate key |
| stock_code | TEXT | natural key, unique |
| description | TEXT | first-seen description for the stock code |

### `warehouse.dim_geography`
| Column | Type | Notes |
|---|---|---|
| geography_key | SERIAL PK | surrogate key |
| country | TEXT | natural key, unique |

### `warehouse.fact_sales`
| Column | Type | Notes |
|---|---|---|
| sales_key | SERIAL PK | |
| invoice_no, stock_code | TEXT | `UNIQUE(invoice_no, stock_code)` — the dedup/idempotency key for incremental loads |
| date_key, product_key, geography_key | FK | NOT NULL |
| customer_key | FK | nullable (guest orders) |
| quantity | INTEGER | |
| unit_price, revenue | NUMERIC | `revenue = quantity * unit_price` |

### `warehouse.fact_customer_retention`
| Column | Type | Notes |
|---|---|---|
| retention_key | SERIAL PK | |
| customer_key | FK | |
| snapshot_date | DATE | date the RFM snapshot was computed |
| recency_days | INTEGER | days since last purchase, relative to snapshot_date |
| frequency | INTEGER | distinct invoices |
| monetary | NUMERIC | total revenue |
| rfm_segment | TEXT | `Champions` / `Loyal` / `At Risk` / `Lost`, from quartile-scored R+F+M (see `src/features/rfm_features.py`) |

## Marts (`marts` schema, consumed by the dashboard)

- **`marts.sales_mart`** — one row per `fact_sales` line, denormalized with date/product/geography attributes.
- **`marts.customer_mart`** — one row per customer, denormalized with their latest retention snapshot.
