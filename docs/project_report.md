# E-Commerce Sales Analytics & Customer Churn Prediction

**Project Report — Part 1: Data Pipeline · Part 2: MLOps Extension**

Course: Data Engineering and MLOps | Individual Project

---

## 1. Problem Understanding & Objectives

Online retailers generate transaction-level data that, on its own, is too raw to support business decisions. This project builds an analytics platform that consolidates online retail transactions, customer details, and product information into a form that can answer two kinds of questions: *what is happening to sales* (revenue trends, top products, geographic mix) and *what is happening to customers* (who is buying repeatedly, who has gone quiet, who is at risk of churning).

The project is split into two parts. **Part 1** (sections 1–10) builds the foundation: a reliable retail ETL pipeline, a dimensional data warehouse, and an interactive business dashboard. **Part 2** (sections 11–18) extends the same warehouse with a churn prediction model operationalized through MLOps practices: MLflow tracking and registry, a FastAPI prediction service, Docker packaging, drift monitoring and a retraining policy.

Concretely, Part 1 set out to:

- Build a repeatable ingestion process that keeps a raw, auditable copy of every extraction and records what happened during that extraction (source, timestamp, row count, success/failure).
- Design customer, product, geography, and date dimensions alongside a sales fact table, structured so that re-running the pipeline over the same or overlapping data never produces duplicates.
- Compute recency/frequency/monetary (RFM) customer features and a retention fact table as the analytical bridge into Part 2's churn model.
- Surface all of the above through a dashboard with at least five meaningful, real-data-backed views.
- Demonstrate the entire path — from a cold start to a populated warehouse and a working dashboard — as one successful, reproducible run.

## 2. Data Source

**Primary dataset: UCI Online Retail Dataset**
`https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx`

This is transaction-level data from a UK-based online retailer, covering 01-Dec-2010 through 09-Dec-2011: 541,909 rows with invoice number, stock code, product description, quantity, invoice date, unit price, customer ID, and country. It is free, publicly hosted by UCI, and requires no registration or API key — it is downloaded directly by [`src/ingestion/download_dataset.py`](../src/ingestion/download_dataset.py) at pipeline run time, so no manual data-gathering step is needed and the source is fully reproducible.

**Why no Kaggle source.** The assignment brief lists Kaggle e-commerce datasets as an *optional* alternative source, not a requirement. Introducing a second, differently-shaped dataset would have added integration complexity (reconciling schemas, currencies, and customer identifiers across two unrelated sources) without adding to the pipeline's design demonstration. It was deliberately not used.

**How the "incremental source" requirement was met instead.** The brief's suggested-architecture table calls for "CSV/Excel order files — simulated monthly extracts" specifically to exercise incremental ingestion. Rather than fabricating a second dataset for this, [`src/ingestion/split_into_monthly_extracts.py`](../src/ingestion/split_into_monthly_extracts.py) slices the UCI dataset — which already spans thirteen calendar months — into one CSV per month (`2010-12.csv` through `2011-12.csv`). The incremental loader then ingests these thirteen files one at a time, tracking which have already been processed. This gives a genuine multi-batch, multi-run ingestion story using a single, real, license-clean dataset instead of blending two.

## 3. Architecture

The pipeline follows the layered design suggested in the brief — source, ingestion, raw/staging, transformation, storage, analytics, and (for Part 2) an MLOps layer — implemented concretely as shown below. The full diagram, including the Part 2 layer, is in [`docs/architecture_diagram.md`](architecture_diagram.md).

```
UCI Online Retail dataset
        │  download_dataset.py
        ▼
Simulated monthly CSV extracts  ──►  incremental_loader.py  ──►  data/raw/<run_timestamp>/  (immutable raw copies)
                                                              └─► logs/extraction_log/extraction_log.jsonl (extraction metadata)
        │
        ▼
clean_transform.py  ──► standardize + validate ──┬─► valid rows  → data/staging/cleaned_sales.csv
                                                  └─► invalid rows → data/rejected/*.csv + staging.rejected_records
        │
        ▼
load_warehouse.py  ──► PostgreSQL star schema (warehouse.dim_*, warehouse.fact_sales)
        │
        ▼
rfm_features.py  ──► warehouse.fact_customer_retention
        │
        ▼
marts.sales_mart / marts.customer_mart  ──►  Streamlit dashboard
```

Every step from ingestion through RFM computation runs as a task in a single Airflow DAG (`ecommerce_etl_dag`), so the layered flow above is also the literal task graph a grader can see execute in the Airflow UI.

**Runtime.** PostgreSQL and Airflow run via Docker Compose ([`docker-compose.yml`](../docker-compose.yml)) so the whole backend starts with one command and is identical on any machine. The Streamlit dashboard runs directly with Python against the same Postgres instance, which keeps the interactive-dashboard iteration loop fast and matches the brief's suggestion to reach for Streamlit "when interactive filtering, user input, or future model prediction is central to the application" — relevant here because Part 2's live churn predictions are integrated into this same dashboard (which, from Part 2 onwards, is also packaged as a container).

## 4. Data Ingestion

Ingestion is deliberately split into small, single-purpose scripts under [`src/ingestion/`](../src/ingestion/), each satisfying one of the brief's minimum ingestion expectations:

| Requirement | How it's met |
|---|---|
| Maintain a raw copy of the extracted data | `ingest_raw.py` copies every source file into `data/raw/<UTC-run-timestamp>/` before anything else touches it. This copy is never modified by later stages. |
| Record extraction date, source, file/API status, and row count | Every ingestion attempt — success or failure — appends one JSON line to `logs/extraction_log/extraction_log.jsonl` with `extraction_date`, `source`, `source_file`, `status`, `row_count`, `raw_copy`, and `error`. |
| Handle extraction errors and log failed records | `ingest_file()` wraps the read/copy in a try/except; a missing or unreadable file is logged with `status: "failed"` and the exception message, rather than crashing the run. |
| Support at least one repeatable or scheduled ingestion workflow | `incremental_loader.py` tracks already-ingested monthly files in `logs/extraction_log/processed_files.json`; re-running it only ingests files not seen before, and it also runs as a scheduled Airflow task (`@monthly`). |
| Do not manually edit the final analytical dataset | No step in the pipeline writes to the warehouse by hand; every warehouse row is written by `load_warehouse.py` from the staging CSV. |

**Verified behavior.** Running the loader against the thirteen monthly extracts ingests all thirteen and logs thirteen `"success"` entries. Running it again immediately afterward logs `Incremental ingestion complete: 0 new file(s) processed` — confirming the "repeatable without duplicating work" property the brief asks for.

## 5. ETL & Data Quality

[`src/etl/clean_transform.py`](../src/etl/clean_transform.py) standardizes column names and types (renaming `InvoiceNo` → `invoice_no`, etc.; parsing dates; coercing quantity/price to numeric; rounding price to 2 decimals since the source is entirely GBP with no currency conversion needed) and writes the standardized-but-not-yet-validated rows into `staging.sales_raw` in Postgres — the raw/staging DB layer the brief's architecture calls for.

[`src/etl/data_quality_checks.py`](../src/etl/data_quality_checks.py) then applies seven ordered validation rules (full list in [`docs/data_dictionary.md`](data_dictionary.md)) covering missing keys, cancelled orders (invoice numbers prefixed `C`), non-positive quantity/price, unparseable dates, and missing country. A row failing any rule is routed to `data/rejected/rejected_<batch_id>.csv` and `staging.rejected_records` (as JSON, with the specific reason attached) instead of the warehouse.

**Results from the full dataset run** (541,909 input rows, one batch across all thirteen monthly extracts):

| Outcome | Rows | % |
|---|---:|---:|
| Valid → loaded downstream | 530,100 | 97.8% |
| Rejected: cancelled order | 9,288 | 1.7% |
| Rejected: non-positive quantity | 1,336 | 0.2% |
| Rejected: non-positive unit price | 1,185 | 0.2% |
| **Total rejected** | **11,809** | **2.2%** |

No rows were rejected for missing invoice number, stock code, or country — the UCI dataset is complete on those fields; the actual data-quality issues in this dataset are cancellations and a handful of zero-price/negative-quantity adjustment lines, exactly the cases the brief's "remove cancelled orders and invalid quantities" instruction anticipates.

## 6. Data Warehouse Design

The warehouse is a conventional star schema in PostgreSQL, defined in [`src/db/schema/`](../src/db/schema/) and fully documented column-by-column in [`docs/data_dictionary.md`](data_dictionary.md):

- **`warehouse.dim_date`** — one row per calendar date, `YYYYMMDD` surrogate key, with month/quarter/year/weekday breakdowns for the dashboard's time-based charts.
- **`warehouse.dim_customer`** — one row per known `CustomerID`; guest/unidentified purchases (blank `CustomerID` in the source) are intentionally *not* given a dimension row, so their sales still load into the fact table with `customer_key = NULL` rather than being dropped or faked.
- **`warehouse.dim_product`**, **`warehouse.dim_geography`** — natural-keyed on stock code and country respectively.
- **`warehouse.fact_sales`** — one row per invoice line, `UNIQUE(invoice_no, stock_code)`.
- **`warehouse.fact_customer_retention`** — one row per customer per RFM snapshot date.

**Incremental loading and duplicate prevention.** [`src/etl/load_warehouse.py`](../src/etl/load_warehouse.py) never truncates or blindly re-inserts. Every dimension upsert uses `ON CONFLICT (<natural key>) DO NOTHING`, and the fact table upserts on its `UNIQUE(invoice_no, stock_code)` constraint with `ON CONFLICT ... DO UPDATE`. This makes the load idempotent by construction: reprocessing the same or overlapping staging data any number of times converges to the same warehouse state rather than accumulating duplicates.

**This was verified directly, not assumed.** The Airflow DAG was triggered twice in succession against the full dataset. Both runs produced **identical** results:

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| `fact_sales` rows | 519,599 | 519,599 |
| Total revenue | £10,588,919.89 | £10,588,919.89 |
| Unique customers | 4,338 | 4,338 |
| Unique products | 3,921 | 3,921 |
| Countries | 38 | 38 |

**RFM features.** [`src/features/rfm_features.py`](../src/features/rfm_features.py) computes, per customer, recency (days since last purchase relative to one day after the latest invoice date in the data), frequency (distinct invoices), and monetary value (total revenue), quartile-scores each dimension, and maps the combined score to one of four segments:

| Segment | Customers |
|---|---:|
| Champions | 1,675 |
| At Risk | 1,000 |
| Loyal | 858 |
| Lost | 805 |

This segmentation, and the underlying `fact_customer_retention` table, is the direct analytical predecessor of the Part 2 churn label.

## 7. Orchestration

[`airflow/dags/ecommerce_etl_dag.py`](../airflow/dags/ecommerce_etl_dag.py) chains five tasks: `download_and_split → incremental_ingest → clean_transform → load_warehouse → rfm_features`, scheduled `@monthly` with `catchup=False` and — importantly — `max_active_runs=1`, so a manually triggered run and the automatic scheduled run can never execute concurrently against the same shared files and staging tables (see §9 for why this guard exists).

Postgres and Airflow are brought up together with `docker compose up -d` ([`docker-compose.yml`](../docker-compose.yml)); the Postgres init script ([`docker/postgres-init/01-init-ecommerce-db.sh`](../docker/postgres-init/01-init-ecommerce-db.sh)) creates the `ecommerce` warehouse database and applies the full schema on first start. The DAG is paused by default on a fresh deploy (standard Airflow behavior) and is unpaused and triggered explicitly — from the UI, or with `airflow dags unpause` / `airflow dags trigger` — as documented in the [README](../README.md).

*[Insert screenshot: Airflow UI, `ecommerce_etl_dag` graph view showing all five tasks green after a successful run — see `docs/screenshots/`.]*

## 8. Dashboard

The Streamlit dashboard ([`dashboard/`](../dashboard/)) reads exclusively from the two analytical marts (`marts.sales_mart`, `marts.customer_mart`), never from raw or staging tables, and provides the five required views plus a home page of headline KPIs:

| Page | Content |
|---|---|
| Home ([`app.py`](../dashboard/app.py)) | Total revenue, orders, unique customers, date range; monthly revenue trend |
| Revenue & Order Trends | Daily and monthly revenue line/bar charts |
| Top Products & Categories | Top-N products by revenue, adjustable via a slider, with a supporting table |
| Customer Segmentation | RFM segment pie chart; recency-vs-monetary scatter, sized by frequency and colored by segment |
| Country-wise Sales | Choropleth map of revenue by country; top-15 countries bar chart |
| Retention & Repeat-Purchase Analysis | Repeat-purchase rate metric (65.6% of customers placed more than one order); order-frequency histogram; per-segment summary table |

All five pages were confirmed rendering against the live, fully-loaded warehouse (not sample/mock data) during development.

*[Insert screenshots: each of the six dashboard pages — see `docs/screenshots/`.]*

## 9. End-to-End Execution Evidence

A full run was executed and verified from a cold start:

1. **Fresh environment**: `docker compose down -v` to remove all containers and volumes, then `docker compose up -d` — a genuinely empty Postgres and a freshly-registered, paused Airflow DAG.
2. **Unpause and trigger**: `airflow dags unpause ecommerce_etl_dag` followed by `airflow dags trigger ecommerce_etl_dag`.
3. **Observed run**: `download_and_split` (downloads the dataset, writes 13 monthly CSVs) → `incremental_ingest` (ingests all 13, writing raw copies and extraction-log entries) → `clean_transform` (530,100 valid / 11,809 rejected) → `load_warehouse` (519,599 fact rows, 4,338 customers, 3,921 products) → `rfm_features` (4,338 customers scored). Total DAG run time: **~8.5 minutes**. Final state: `success`.
4. **Second trigger for idempotency**: the DAG was triggered a second time without resetting any state. It correctly found zero new files to ingest, and `load_warehouse`/`rfm_features` reproduced the exact same row counts and revenue total shown in §6 — confirming the pipeline is safe to re-run.
5. **Dashboard**: `streamlit run dashboard/app.py` against the same warehouse, all five analytical pages plus the home page rendering correctly with the numbers above.

Three issues were caught and fixed during this verification rather than glossed over, which is itself part of the pipeline's data-quality story:

- A NaN value in a rejected row broke JSON serialization into `staging.rejected_records` (fixed by explicitly nulling `NaN`/`NaT` before `json.dumps`).
- `pandas.to_sql()`/`pandas.read_sql()` are incompatible with the older SQLAlchemy version bundled by Airflow 2.9, raising `'Engine' object has no attribute 'cursor'`; replaced with direct `psycopg2`-based bulk insert/query helpers ([`src/db/upsert.py`](../src/db/upsert.py)) that don't depend on pandas' version-sensitive engine detection.
- The DAG initially had no concurrency guard: unpausing it fires Airflow's own due scheduled run, which raced against a manual trigger over the same shared files. Fixed with `max_active_runs=1`.

*[Insert screenshots/logs: `docker compose ps` showing healthy containers; `airflow dags list-runs` showing two successful runs; terminal output of the row-count verification queries — see `docs/screenshots/`.]*

## 10. Setup & Reproducibility (Parts 1 and 2)

Full setup and run instructions — prerequisites, `docker compose up`, triggering the DAGs, running the dashboard, the Part 2 ML commands and the tests — are in [`README.md`](../README.md). In summary: `docker compose up -d` brings up Postgres (port 5433, remapped from the default 5432 to avoid clashing with any locally installed Postgres) and Airflow (UI on port 8080, `admin`/`admin`); the DAG is unpaused and triggered once; `streamlit run dashboard/app.py` serves the dashboard on port 8501.

# Part 2 — MLOps Extension

## 11. Overview and Objectives

Part 2 turns the Part 1 warehouse into an operational churn-prediction system. The objectives, taken from the brief, were to define churn with a clear inactivity window; train Logistic Regression, Random Forest and XGBoost models on a chronological split; track experiments with MLflow; register and version the best model; serve it through a FastAPI service; containerise the API and dashboard; integrate predictions into the dashboard; monitor input data, drift, performance, latency and failures; and define retraining criteria and the model lifecycle (see [`docs/model_lifecycle.md`](model_lifecycle.md)). The updated architecture, including the MLOps layer, is in [`docs/architecture_diagram.md`](architecture_diagram.md).

## 12. Churn Definition and Feature Engineering

**Definition.** A customer churns at a cutoff date if they had purchased on or before it and make no purchase in the following **90 days**. Guest checkouts have no customer identity and are excluded.

**Avoiding leakage.** The existing Part 1 RFM features snapshot *after* the last purchase date, so they have no future to be labelled against and would leak if reused. Part 2 builds features **point-in-time**: for each cutoff, features use only purchases on or before it and the label uses only the 90 days after it ([`ml/features/build_features.py`](../ml/features/build_features.py)). Tests prove that adding later purchases never changes a cutoff's features and that a label window running past the end of the data is rejected.

**Features (14):** recency, frequency, monetary value, tenure, average order value, average days between orders, orders and revenue in the last 30/60/90 days, distinct products, and a UK flag. They are stored per customer per cutoff in `warehouse.ml_customer_features`.

**Chronological split** (the model is always tested on the future relative to its training data):

| Split | Cutoffs | Rows | Churn rate |
|---|---|---:|---:|
| Train | 2011-03-01, 2011-04-01 | 3,853 | 43.6% / 46.1% |
| Validation | 2011-06-09 | 2,800 | 52.3% |
| Test | 2011-09-09 | 3,370 | 43.1% |

## 13. Model Development and Evaluation

Three families were trained over small grids (Logistic Regression with scaling, Random Forest, XGBoost), with fixed seeds. The best configuration of each, on the validation split:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** (300 trees, depth 12) | 0.707 | 0.702 | 0.704 | 0.751 | **0.729** |
| XGBoost | 0.721 | 0.675 | 0.697 | 0.757 | 0.723 |
| Logistic Regression | 0.699 | 0.721 | 0.710 | 0.748 | 0.716 |

The model was selected on validation PR-AUC, then evaluated **once** on the untouched test split: **precision 0.599, recall 0.710, F1 0.650, ROC-AUC 0.722, PR-AUC 0.619**.

**Interpretation.** These are moderate, realistic results for a 90-day churn problem on twelve months of retail data, and they should not be read as a strong predictor:

- The three model families are within 0.013 PR-AUC of each other, so the choice of Random Forest is narrow.
- The forest overfits (train F1 0.88 versus validation 0.70).
- Precision dropped from 0.71 (validation) to 0.60 (test) while recall held. The churn base rate fell from 52% to 43% between the two periods, which lowers precision by itself. With only four labelled cutoffs, the validation and test windows are single dates, so the metrics carry real sampling noise.
- The most influential features are monetary value (0.14), 90-day revenue (0.12), average order value (0.10) and distinct products (0.10).

Sanity checks on real customers agree with intuition: a top-spending, recently active customer scores 0.3% churn risk, and a customer with a single purchase in January scores 76%.

## 14. MLflow Tracking, Registry and Versioning

An MLflow server runs in Docker (port 5000) with a PostgreSQL backend store and an artifact volume. Every training execution logs, per run, its parameters, train and validation metrics, a confusion matrix and feature importance, nested under a `model-selection` parent run that also holds the test metrics and the training cutoffs. The selected model is logged with an input/output signature and registered as **`churn-model`**; the alias **`champion`** marks the production version. Consumers load `models:/churn-model@champion`, so promoting a new version needs no code change. Two versions are registered (v1, v2); retraining on identical data produced identical metrics, which demonstrates reproducibility.

*[Insert screenshots: MLflow experiment run table with the three model families; the `churn-model` registry page showing the champion alias — see `docs/screenshots/`.]*

## 15. Prediction API and Docker Deployment

The FastAPI service ([`api/main.py`](../api/main.py)) exposes `GET /health`, `POST /predict` (14 validated features) and `POST /predict/customer/{id}` (features built from the warehouse as of the latest data date). Every request is written to `warehouse.prediction_log` with its features, probability, latency and status; a logging failure never breaks a prediction.

Verified behaviour with real requests: a valid payload returns HTTP 200; missing, negative, unknown or out-of-range fields return 422; an unknown customer returns 404; a SQL-injection-style customer id returns 404 (ids are bound parameters, never concatenated); a missing model returns 503 and a scoring failure returns 500 and is logged. Average serving latency was about 270 ms with no errors.

The API and the dashboard each have a pinned Dockerfile and a service in `docker-compose.yml`, alongside Postgres, MLflow and Airflow, so one `docker compose up -d` starts the whole system. The containerised API loaded champion v2 through MLflow and returned exactly the same probability (0.7644) as the same customer scored from the host environment, confirming the training and serving environments agree.

*[Insert screenshot: API Swagger page at `http://localhost:8000/docs` and a successful `/predict/customer/12346` response.]*

## 16. Monitoring and Retraining

Monitoring ([`ml/monitoring/drift_monitor.py`](../ml/monitoring/drift_monitor.py)) stores every check in `warehouse.monitoring_metrics`: input data quality (null and out-of-range rates), feature drift (Population Stability Index against the data the champion was trained on), model precision and recall against the champion's recorded test recall, and serving error rate and p50/p95 latency from the prediction log.

**Retraining criteria** (`ml/monitoring/retraining.py`): retrain if at least 2 features have PSI above 0.20, or recall falls more than 0.10 below baseline, or the serving error rate exceeds 2%. A **guard** defers retraining unless labelled cutoffs newer than the champion's training data exist; retraining on the same data would only recreate the same model and would loop forever while drift persists. When new data exists, the train/validation/test windows roll forward and the drift reference moves with them.

**What monitoring found.** On the current data 7 of 14 features exceed the PSI threshold (largest: tenure 2.28, recency 0.69, 90-day orders 0.66). The drift is genuine and explained: cumulative features grow as the data window grows, and the model trained on customers with 1–4 months of history now scores customers with up to 12. The policy therefore recommends retraining, and the pipeline correctly defers it because the data ends on 2011-12-09, so no labelled cutoff newer than the one already used for testing exists. A simulated-drift mode (`--simulate-drift`) exercises the breach path (9 of 14 features breach), and unit tests cover the decision rules.

## 17. Dashboard Integration

A **Churn Risk** page was added to the Streamlit dashboard. It shows API and champion-model status, the latest monitoring breaches (7 of 35 checks), the customers-by-risk-tier chart (High 901, Medium 1,695, Low 1,742), the churn-probability distribution, risk tiers within each RFM segment, the highest-value customers at high risk (£314,128 of revenue sits in the high-risk tier), and a live single-customer lookup that calls the containerised API. Batch scores come from `ml/scoring/score_customers.py`, which writes `warehouse.customer_churn_scores` and runs as the last task of the ML DAG.

*[Insert screenshots: the Churn Risk page (KPIs and charts) and a live customer lookup — see `docs/screenshots/`.]*

## 18. Part 2 Execution Evidence

The ML pipeline runs as the Airflow DAG `ml_pipeline_dag`: `build_features → monitor → decide_retraining → (retrain_model | skip_retraining) → score_customers` (weekly, paused by default, `max_active_runs=1`). It was run end to end in Airflow with every task successful, and the Part 1 `ecommerce_etl_dag` was re-run afterwards on the rebuilt Airflow image and reproduced the identical warehouse (519,599 fact rows, £10,588,919.89 revenue). The automated test suite has 33 tests (Part 1 and Part 2) and all pass.

Problems found and fixed during verification, rather than assumed away:

- MLflow's emoji run URLs crashed the Windows console; output is now forced to UTF-8.
- MLflow's safe model serialisation rejected the random forest; only scikit-learn's `Tree` type is trusted, not everything.
- The Airflow image was on Python 3.11, which cannot install the XGBoost version the model was trained with; it now uses the Python 3.12 Airflow build with pinned versions, and pandas was aligned so `pip check` passes cleanly.
- The Airflow image was missing `skops`, so the first ML DAG run failed at the monitor task; it was added and the DAG re-run successfully.
- Retraining originally had no notion of "new data", so drift would have triggered an endless loop of identical retrains; the new-data guard and rolling windows fixed this.

*[Insert screenshots: Airflow `ml_pipeline_dag` graph with all tasks green; `docker compose ps` showing all services healthy.]*

## Appendix: Data Dictionary

See [`docs/data_dictionary.md`](data_dictionary.md) for the full source-column reference, the seven ordered validation rules, and the complete warehouse/mart schema. Key summary figures from the verified run:

- **Source**: UCI Online Retail Dataset, 541,909 rows, 01-Dec-2010 to 09-Dec-2011.
- **After cleaning**: 530,100 valid rows (97.8%), 11,809 rejected (2.2% — cancellations, non-positive quantity/price).
- **Warehouse**: 519,599 fact rows, 4,338 customers, 3,921 products, 38 countries, £10,588,919.89 total revenue.
- **RFM segments**: Champions 1,675 · At Risk 1,000 · Loyal 858 · Lost 805.
- **Top countries by revenue**: United Kingdom (£8.96M), Netherlands (£285K), EIRE (£281K), Germany (£228K), France (£210K).
- **Part 2**: 4 labelled cutoffs, 14 point-in-time features; champion Random Forest test F1 0.650 (precision 0.599, recall 0.710, ROC-AUC 0.722); 4,338 customers scored (901 High / 1,695 Medium / 1,742 Low risk); 7 of 14 features flagged for drift, retraining deferred for lack of newer labelled data.
