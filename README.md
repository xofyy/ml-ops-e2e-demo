# FinanceMath MLOps Demo

FinanceMath MLOps Demo is a lightweight, end-to-end pipeline built around the [FinanceMath](https://huggingface.co/datasets/yale-nlp/FinanceMath) dataset. It covers data ingestion, feature engineering, model training/evaluation, inference serving, orchestration, and monitoring in a reproducible manner.

## Features
- **Ingestion** – Download the validation split from Hugging Face, parse markdown tables, validate each record with Pydantic, and persist JSONL artefacts.
- **Feature engineering** – Generate sentence embeddings, numeric aggregates from tables, and metadata features stored in Parquet.
- **Training & tracking** – Train a LightGBM regressor while logging metrics and artefacts to MLflow.
- **Evaluation** – Reuse the feature set to compute MAE/RMSE and store a JSON report.
- **Serving** – FastAPI service loads the latest MLflow model and exposes Prometheus metrics.
- **Orchestration** – Prefect flow connects ingestion › features › train › evaluate; deployments are available in Prefect Cloud and CI.
- **Monitoring** – Evidently-based drift report, Prometheus + Grafana dashboards, Docker Compose stack.

## Quick Start
```bash
# (Optional) create and activate a conda environment
conda create -n finance-math python=3.10
conda activate finance-math

# Authenticate with Hugging Face if the dataset is gated
huggingface-cli login  # or export HUGGINGFACE_TOKEN=<your-token>

# Install project dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Update configs/inference.yaml to point to your trained model (e.g. model.model_uri: "runs:/<run_id>/model")

# Run the pipeline end to end
python -m src.data.ingest
python -m src.features.build_features
RUN_ID=$(python -m src.models.train)
python -m src.models.evaluate --run-id "$RUN_ID"
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# Metrics endpoint
curl http://localhost:8000/metrics
```

## Make Targets
| Command | Description |
|---------|-------------|
| `make install` | Install dependencies via pip. |
| `make data` | Run ingestion step. |
| `make features` | Build feature parquet. |
| `make train` | Train LightGBM model with MLflow tracking. |
| `make evaluate RUN_ID=<id>` | Evaluate a specific MLflow run. |
| `make serve` | Start FastAPI inference server. |
| `make prefect-flow` | Execute Prefect flow locally. |
| `make lint` / `make lint-fix` | Static analysis with Ruff & Black. |
| `make test` | Run unit tests. |
| `make drift-report MODEL_URI=runs:/<run_id>/model` | Generate Evidently HTML drift report. |
| `make compose-up` / `make compose-down` | Bring up or tear down Docker Compose stack (inference + Prometheus + Grafana). |
| `SKIP_MODEL_LOAD=1 pytest` | Skip model loading during tests (FastAPI metrics tests). |

## Repository Layout
```
.
+¦¦ configs/              # YAML configs for data, training, inference.
+¦¦ data/                 # Data artefacts (tracked via DVC).
+¦¦ docker/               # Dockerfiles and compose stack for serving / monitoring.
+¦¦ docs/                 # Architecture notes and notebooks.
+¦¦ scripts/              # Utility scripts (e.g. Evidently report).
+¦¦ src/
-   +¦¦ common/           # Shared config loaders and schemas.
-   +¦¦ data/             # Ingestion logic & validators.
-   +¦¦ features/         # Feature engineering pipeline.
-   +¦¦ models/           # Training and evaluation.
-   +¦¦ serving/          # FastAPI app and predictor wrapper.
-   L¦¦ workflows/        # Prefect orchestration.
L¦¦ tests/                # Pytest unit tests.
```

## Data Validation & Versioning
- Ingestion stage validates each JSONL record using `src/data/validators.py`; invalid rows are logged and skipped.
- Track data snapshots with DVC:
  ```bash
  dvc init
  dvc remote add -d storage gs://<bucket>/finance-math
  dvc add data/raw data/processed
  git add data/raw.dvc data/processed.dvc .dvc/config
  dvc push
  ```
  Log `.dvc` hashes or `dvc status -c` output to MLflow runs for data/model traceability. Use `dvc pull` on other machines to restore the same snapshot.

## Docker Compose
Launch inference, Prometheus, and Grafana together:
```bash
cd docker
docker compose up --build
```
- Inference service: `http://localhost:8000` (metrics at `/metrics`).
- Prometheus: `http://localhost:9090`.
- Grafana: `http://localhost:3000` (default credentials `admin` / `admin`), with the “FinanceMath Inference Overview” dashboard showing request counts, latency (p95), and prediction totals.

## Prefect CI Trigger
- Manual run via GitHub Actions: `Actions ? prefect-run ? Run workflow`.
- Define `PREFECT_API_KEY` and `PREFECT_WORKSPACE` (e.g. `account_id/workspace_id`) as repository secrets so the workflow can log into Prefect Cloud and trigger the `finance-math-mlops/finance-math-demo` deployment.

## CI/CD
- `.github/workflows/mlops-demo.yml` installs dependencies (`requirements-ci.txt`), runs lint/test, builds the inference Docker image, and optionally generates a drift report (skipped if the Parquet features are absent).
- `.github/workflows/prefect-run.yml` manually triggers a Prefect deployment run (requires Prefect Cloud credentials).

## Next Steps
- Schedule Prefect deployments or trigger them automatically from GitHub Actions.
- Point MLflow tracking to a shared backend (PostgreSQL, Databricks, etc.).
- Improve model performance via hyperparameter search or alternative algorithms.
- Integrate Prometheus metrics with Grafana alerts / third-party monitoring.
