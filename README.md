# E-Commerce Sales Analytics & Customer Churn Prediction

**Part 1** is a retail ETL pipeline (Python/Pandas + Airflow) that loads the
UCI Online Retail dataset into a PostgreSQL star schema, plus a Streamlit
dashboard for sales and retention analytics. **Part 2** adds customer-churn
prediction with MLOps: point-in-time features, MLflow experiment tracking and
model registry, a FastAPI prediction service, drift/performance monitoring
with a retraining policy, and a churn-risk dashboard page, all containerised.

See [`docs/architecture_diagram.md`](docs/architecture_diagram.md) for the
full data-flow diagram and [`docs/data_dictionary.md`](docs/data_dictionary.md)
for table definitions and validation rules, and
[`docs/model_lifecycle.md`](docs/model_lifecycle.md) for the churn model's
definition, evaluation, monitoring and retraining rules.

## Prerequisites

- Docker Desktop (Postgres, Airflow, MLflow, API and dashboard containers)
- Python 3.11+ (for running the dashboard and/or scripts locally)

## 1. Start Postgres + Airflow

```bash
docker compose up -d
```

- Postgres is reachable at `localhost:5433` (mapped off the default 5432 to
  avoid clashing with any locally installed Postgres service; databases:
  `airflow` for Airflow's own metadata, `ecommerce` for the warehouse —
  created automatically on first start by
  `docker/postgres-init/01-init-ecommerce-db.sh`, which also applies the
  schema in `src/db/schema/`).
- Airflow UI: http://localhost:8080 (user `admin` / password `admin`, created
  by the `airflow-init` service).

## 2. Run the pipeline

The DAG is paused by default (standard Airflow behavior on first deploy).
In the Airflow UI, toggle `ecommerce_etl_dag` on, then click "Trigger DAG" —
or from a shell: `docker compose exec airflow-scheduler airflow dags unpause ecommerce_etl_dag`
followed by `docker compose exec airflow-scheduler airflow dags trigger ecommerce_etl_dag`.

(Unpausing and triggering separately, rather than starting pre-unpaused,
avoids the scheduler also firing its own automatic `@monthly` catch-up run
at the same time as your manual trigger, which would race both runs over
the same files/tables.)

Once triggered, the DAG will:

1. Download the UCI Online Retail dataset and slice it into simulated
   monthly extracts.
2. Ingest new monthly extracts incrementally, keeping a raw copy and an
   extraction log entry for each.
3. Clean and validate the data, routing invalid rows to `data/rejected/`.
4. Load the star schema in Postgres (idempotent upserts).
5. Compute RFM features into the customer-retention fact table.

Re-running the DAG only ingests any *new* monthly extracts and safely
re-upserts existing warehouse rows without duplicating them.

### Running the pipeline steps locally instead of via Airflow

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # adjust POSTGRES_* / DATABASE_URL if needed

python -m src.ingestion.download_dataset
python -m src.ingestion.split_into_monthly_extracts
python -m src.ingestion.incremental_loader
python -m src.etl.clean_transform data/raw/**/*.csv   # or pass specific raw file paths
python -m src.etl.load_warehouse
python -m src.features.rfm_features
```

## 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at http://localhost:8501 with 5 pages: revenue trends, top products,
customer segmentation, country-wise sales, and retention analysis.

## 4. Part 2 — churn model (MLflow, API, monitoring)

`docker compose up -d` (step 1) also starts, in addition to Postgres and Airflow:

| Service | URL | Purpose |
|---|---|---|
| MLflow | http://localhost:5000 | experiment tracking + model registry |
| Prediction API | http://localhost:8000 (`/docs` for Swagger) | `/health`, `/predict`, `/predict/customer/{id}` |
| Dashboard | http://localhost:8501 | all Part 1 pages + **Churn Risk** |

Run the ML steps from the local virtualenv (`pip install -r requirements.txt`,
`.env` as above), in this order. The first run needs the Part 1 warehouse to be
loaded:

```bash
python -m ml.features.build_features      # point-in-time features + 90-day churn labels
python -m ml.training.train_model         # trains 3 model families, registers churn-model@champion
python -m ml.scoring.score_customers      # batch churn scores for every customer
python -m ml.monitoring.drift_monitor     # drift / quality / performance checks (add --simulate-drift to demo)
```

Restart the API after registering a new champion (`docker compose restart api`)
so it loads the new version.

Or let Airflow run the loop: unpause `ml_pipeline_dag` in the Airflow UI
(`build_features -> monitor -> decide_retraining -> retrain_model | skip_retraining
-> score_customers`; weekly, paused by default). Retraining is only performed
when monitoring justifies it *and* newer labelled data exists.

Try the API:

```bash
curl -X POST http://localhost:8000/predict/customer/12346
```

Run the tests (unit tests; no services needed):

```bash
python -m pytest tests -q
```

## Project layout

See [`docs/architecture_diagram.md`](docs/architecture_diagram.md) for the
data-flow picture; the directory-by-directory breakdown lives in this
project's plan file and in the module docstrings under `src/`.

## Environment variables

Copy `.env.example` to `.env` and adjust as needed. Real credentials are
never committed — only `.env.example` is tracked in git.
