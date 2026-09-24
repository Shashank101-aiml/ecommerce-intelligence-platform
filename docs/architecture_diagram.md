# Architecture Diagram

## Part 1 — Data pipeline

```mermaid
flowchart TB
    subgraph Source["Source Layer"]
        A1[UCI Online Retail dataset]
        A2["Simulated monthly CSV extracts\n(sliced from the same dataset)"]
    end

    subgraph Ingestion["Ingestion Layer (Python / Airflow)"]
        B1[download_dataset.py]
        B2[split_into_monthly_extracts.py]
        B3[incremental_loader.py]
        B4[ingest_raw.py]
    end

    subgraph RawStaging["Raw / Staging Layer"]
        C1[("data/raw/&lt;run_timestamp&gt;/\nimmutable raw copies")]
        C2[("logs/extraction_log/\nextraction metadata")]
        C3[("staging.sales_raw\n(Postgres)")]
    end

    subgraph Transform["Transformation Layer"]
        D1[clean_transform.py]
        D2[data_quality_checks.py]
        D3[("data/rejected/\n+ staging.rejected_records")]
        D4[("data/staging/cleaned_sales.csv")]
        D5[rfm_features.py]
    end

    subgraph Storage["Storage Layer (PostgreSQL star schema)"]
        E1[(dim_date)]
        E2[(dim_customer)]
        E3[(dim_product)]
        E4[(dim_geography)]
        E5[(fact_sales)]
        E6[(fact_customer_retention)]
    end

    subgraph Analytics["Analytics Layer"]
        F1[(marts.sales_mart)]
        F2[(marts.customer_mart)]
        F3[Streamlit Dashboard]
    end

    A1 --> B1 --> B2 --> A2
    A2 --> B3 --> B4
    B4 --> C1
    B4 --> C2
    D1 --> C3
    C1 --> D1 --> D2
    D2 -- invalid rows --> D3
    D2 -- valid rows --> D4
    D4 --> E1 & E2 & E3 & E4 & E5
    E5 --> D5 --> E6
    E1 & E3 & E4 & E5 --> F1
    E2 & E6 --> F2
    F1 & F2 --> F3
```

**Orchestration:** ingestion, transformation and storage run as tasks in the Airflow DAG `ecommerce_etl_dag` (`airflow/dags/ecommerce_etl_dag.py`), scheduled monthly and triggerable on demand.

## Part 2 — MLOps layer

```mermaid
flowchart TB
    W[("Warehouse\nfact_sales · dim_*")]

    subgraph Features["Feature layer (ml/features)"]
        G1["build_features.py\npoint-in-time features + 90-day churn label\nper cutoff date"]
        G2[("warehouse.ml_customer_features")]
    end

    subgraph Training["Training & registry (ml/training)"]
        H1["train_model.py\nLogReg · RandomForest · XGBoost\nchronological train / validation / test"]
        H2[["MLflow server :5000\nruns · params · metrics · artifacts\n(Postgres backend + artifact volume)"]]
        H3[["Model Registry\nchurn-model · alias: champion"]]
    end

    subgraph Serving["Serving (Docker)"]
        I1["FastAPI :8000\n/health · /predict · /predict/customer/{id}"]
        I2[("warehouse.prediction_log")]
        I3["score_customers.py\nbatch scoring + risk tiers"]
        I4[("warehouse.customer_churn_scores")]
    end

    subgraph Monitoring["Monitoring & retraining (ml/monitoring)"]
        J1["drift_monitor.py\nPSI drift · data quality · precision/recall\nlatency · error rate"]
        J2[("warehouse.monitoring_metrics")]
        J3{"retraining.py\ndrift, recall drop or errors?\nAND new labelled cutoffs?"}
    end

    K["Streamlit dashboard :8501\nChurn Risk page"]

    W --> G1 --> G2 --> H1
    H1 --> H2 --> H3
    H3 -- "load champion" --> I1
    H3 -- "load champion" --> I3
    W --> I1
    I1 --> I2
    I3 --> I4
    G2 --> J1
    I2 --> J1
    H3 --> J1
    J1 --> J2 --> J3
    J3 -- "yes: roll windows forward" --> H1
    I4 --> K
    J2 --> K
    I1 -- "live lookup" --> K
```

**Orchestration:** `ml_pipeline_dag` (`airflow/dags/ml_pipeline_dag.py`, weekly, paused by default) runs
`build_features → monitor → decide_retraining → (retrain_model | skip_retraining) → score_customers`.

**Runtime (`docker compose up -d`):** postgres `:5433` · mlflow `:5000` · api `:8000` · dashboard `:8501` · airflow `:8080`.
