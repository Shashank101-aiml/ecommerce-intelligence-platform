# Architecture Diagram

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

    subgraph MLOps["MLOps Layer (Part 2 — planned)"]
        G1[Feature engineering]
        G2[MLflow tracking + registry]
        G3[FastAPI prediction service]
        G4[Drift / performance monitoring]
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

    F2 -. feeds .-> G1 --> G2 --> G3
    G3 -. drift/perf metrics .-> G4
```

**Orchestration:** all Ingestion + Transformation + Storage steps above run as tasks in the Airflow DAG `ecommerce_etl_dag` (`airflow/dags/ecommerce_etl_dag.py`), scheduled monthly and triggerable on demand.
