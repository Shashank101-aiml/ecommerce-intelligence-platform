# Model Lifecycle — Customer Churn

How the churn model is defined, trained, registered, served, monitored and retrained. All numbers are from the verified run on the UCI Online Retail warehouse (4,338 customers, data from 2010-12-01 to 2011-12-09).

## 1. Churn definition

A customer **churns** at a cutoff date if they had purchased on or before the cutoff and make **no purchase in the following 90 days** (`CHURN_WINDOW_DAYS` in `ml/config.py`). Customers with no purchase before the cutoff are not scored.

Guest checkouts (no `CustomerID`) cannot be tracked over time, so they are excluded from modelling.

## 2. Leakage control and splits

Features for a cutoff use **only purchases on or before it**; the label uses only the 90 days after it (`ml/features/build_features.py`). Unit tests (`tests/test_ml_features.py`) prove that adding later purchases never changes a cutoff's features, and that a label window running past the end of the data is rejected.

The split is chronological, by cutoff date, so the model is always evaluated on the future relative to what it trained on:

| Split | Cutoff dates | Customers | Churn rate |
|---|---|---:|---:|
| Train | 2011-03-01, 2011-04-01 | 3,853 rows | 43.6%, 46.1% |
| Validation | 2011-06-09 | 2,800 | 52.3% |
| Test | 2011-09-09 (label window ends 2011-12-08) | 3,370 | 43.1% |

The test split is evaluated **once**, only for the model already selected on validation.

## 3. Features (14, all computed point-in-time)

`recency_days`, `frequency`, `monetary`, `tenure_days`, `avg_order_value`, `avg_days_between_orders`, `orders_30d/60d/90d`, `revenue_30d/60d/90d`, `distinct_products`, `is_uk`. Stored in `warehouse.ml_customer_features` (one row per customer per cutoff).

## 4. Model selection

Three families, small grids, all tracked in MLflow (experiment `churn-prediction`; every run logs parameters, train and validation metrics, a confusion matrix and feature importance). Best configuration per family, on the validation split:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Random forest** (300 trees, depth 12) | 0.707 | 0.702 | 0.704 | 0.751 | **0.729** |
| XGBoost | 0.721 | 0.675 | 0.697 | 0.757 | 0.723 |
| Logistic regression | 0.699 | 0.721 | 0.710 | 0.748 | 0.716 |

Selection metric: validation **PR-AUC** (churn is close to balanced but the ranking quality is what the retention team uses). The three families are within 0.013 PR-AUC of each other, so the choice of random forest is a narrow one; it also overfits noticeably (train F1 0.88 vs validation 0.70), which is why held-out testing and monitoring matter.

**Champion, evaluated once on the test split:** precision 0.599, recall 0.710, F1 0.650, ROC-AUC 0.722, PR-AUC 0.619.

Validation to test: precision fell from 0.71 to 0.60 while recall held. The churn base rate also fell from 52% to 43% between the two periods, which by itself lowers precision; this is an honest limit of a 12-month dataset, not something tuned away.

Most important features (random forest): `monetary` 0.14, `revenue_90d` 0.12, `avg_order_value` 0.10, `distinct_products` 0.10, `avg_days_between_orders` 0.09.

## 4b. Reproducibility

Seeds are fixed. Retraining on identical data produced identical metrics (registry v1 and v2 match exactly). Dependency versions are pinned in `requirements.txt`, `api/requirements.txt` and the Airflow image so the training and serving environments load the same model.

## 5. Registry and versioning

The selected model is logged with an input/output signature and registered as **`churn-model`** in the MLflow Model Registry. The alias **`champion`** points at the version in production; the API and batch-scoring job load `models:/churn-model@champion`, so promoting a new version needs no code change. Each version is linked to the run that produced it (parameters, cutoffs used, metrics, artifacts). Currently registered: v1 and v2 (champion).

## 6. Serving

- **Online:** FastAPI (`api/main.py`, Docker, port 8000). `POST /predict` takes the 14 features (strict validation: missing, negative, unknown or out-of-range fields return 422); `POST /predict/customer/{id}` builds the features from the warehouse as of the latest data date; `GET /health` reports the loaded model version and returns `degraded` if none loaded. Customer ids are bound as SQL parameters, never interpolated into query text.
- **Audit log:** every request is stored in `warehouse.prediction_log` (features, probability, label, latency, status, error). Logging failures never break a prediction.
- **Batch:** `ml/scoring/score_customers.py` scores every customer into `warehouse.customer_churn_scores` with a risk tier (High ≥ 0.70, Medium ≥ 0.40, else Low). Latest run: 901 High, 1,695 Medium, 1,742 Low.
- **Dashboard:** the Churn Risk page shows API/model status, monitoring breaches, tier charts, the highest-value at-risk customers and a live single-customer lookup.

## 6b. Model output is checked against reality

Sanity checks on the served model: a top-spending, recently active customer (14646) scores 0.3% churn risk; a customer with a single purchase in January (12346) scores 76%. The containerised API and the host returned the identical probability for the same customer (0.7644).

## 7. Monitoring

`ml/monitoring/drift_monitor.py` writes every check to `warehouse.monitoring_metrics`:

| Check | Method | Threshold |
|---|---|---|
| Input data quality | null rate per feature; out-of-range / negative / non-binary values | null > 5%; any out-of-range |
| Feature drift | Population Stability Index of each feature vs. the data the champion was trained on | PSI > 0.20 |
| Model performance | precision, recall on the latest labelled snapshot; recall drop vs. the champion's recorded test recall | drop > 0.10 |
| Serving health | error rate, p50 and p95 latency from `prediction_log` | error rate > 2%; p95 > 1000 ms |

Observed: average serving latency about 270 ms, zero errors. On the current data 7 of 14 features exceed the PSI threshold (largest: `tenure_days` 2.28, `recency_days` 0.69, `orders_90d` 0.66). This drift is real and expected: cumulative features such as tenure grow as the data window grows, and the model was trained on customers with only 1-4 months of history but now scores customers with up to 12 months.

## 8. Retraining criteria

Retrain when **any** of:

1. **Feature drift:** at least 2 features have PSI above 0.20.
2. **Performance decay:** recall falls more than 0.10 below the champion's recorded test recall.
3. **Serving errors:** error rate above 2%.

**Guard:** a retrain is deferred unless the warehouse holds labelled cutoffs newer than the ones the champion trained on (recorded in the run as `data_max_cutoff`). Retraining on the same data would only recreate the same model, and would loop forever whenever drift stays high. When new cutoffs exist, the train / validation / test windows roll forward (newest cutoff = test, the one before = validation, the rest = train) and the new version is registered and becomes champion; the drift reference automatically moves to the new training window.

Current state: monitoring recommends retraining (drift), and the pipeline correctly defers it because the data ends on 2011-12-09, so the latest cutoff that can be labelled with a full 90-day window is 2011-09-10, a day after the one already used for testing. `ml_pipeline_dag` executes this whole loop (`build_features → monitor → decide_retraining → retrain_model | skip_retraining → score_customers`).

## 9. Known limitations

- Twelve months of data: only four labelled cutoffs, so the validation and test windows are single dates and metrics have real sampling noise.
- Performance monitoring uses the latest labelled snapshot; for live predictions the true outcome is only known 90 days later, so live precision/recall lags by that window.
- Promotion is automatic (a retrained model becomes champion). A production setup would add a challenger stage that must beat the champion on the newest test window before the alias moves.
