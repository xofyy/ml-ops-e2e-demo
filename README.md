# FinanceMath MLOps Demo

FinanceMath MLOps Demo is a lightweight, end-to-end pipeline that ingests the [FinanceMath](https://huggingface.co/datasets/yale-nlp/FinanceMath) dataset, engineers features, trains a regression model, evaluates it, and exposes an inference service. The repository highlights reproducible experimentation, orchestration, CI/CD, and monitoring practices.

## Features
- **Ingestion**: Download the Hugging Face validation split, normalise markdown tables, validate each record with Pydantic, and persist JSONL artefacts.
- **Feature engineering**: Combine sentence embeddings with numeric table aggregates and metadata features.
- **Training & tracking**: Fit a LightGBM regressor and capture metrics / artefacts via MLflow.
- **Evaluation**: Reuse the feature set to compute MAE / RMSE and store JSON reports.
- **Serving**: FastAPI service loads the latest MLflow model and exposes Prometheus-compatible metrics.
- **Orchestration**: Prefect flow stitches ingestion -> features -> training -> evaluation.
- **Monitoring**: Evidently script scaffold for drift / performance dashboards.

## Quick Start
```bash
# (Optional) create and activate a conda environment
conda create -n finance-math python=3.10
conda activate finance-math

# Authenticate with Hugging Face if the dataset requires gated access
huggingface-cli login  # or export HUGGINGFACE_TOKEN=<your-token>

# Install project and dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Update configs/inference.yaml to point to your trained model,
#   e.g. set model.model_uri: "runs:/<run_id>/model" or register the model.

# Run the pipeline end to end
python -m src.data.ingest
python -m src.features.build_features
RUN_ID=$(python -m src.models.train)
python -m src.models.evaluate --run-id "$RUN_ID"
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# Metrics endpoint (Prometheus format)
curl http://localhost:8000/metrics
```

Alternatively, Make targets wrap the same commands:
```bash
make data
make features
RUN_ID=$(make -s train)
make evaluate RUN_ID=$RUN_ID
# Update configs/inference.yaml with RUN_ID before serving
make serve
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
| `make drift-report MODEL_URI=runs:/<run_id>/model` | Generate Evidently HTML drift report (computes predictions via MLflow). |
| `make compose-up` / `make compose-down` | Bring up or tear down the Docker Compose stack (inference + Prometheus + Grafana). |
| `SKIP_MODEL_LOAD=1 pytest` | Skip model loading during tests (FastAPI metrics tests). |

## Repository Layout
```
.
├── configs/              # YAML configs for data, training, inference.
├── data/                 # Data artefacts (tracked via DVC).
├── docker/               # Dockerfiles and compose stack for serving / monitoring.
├── docs/                 # Architecture notes and notebooks.
├── scripts/              # Utility scripts (e.g. Evidently report).
├── src/
│   ├── common/           # Shared config loaders and schemas.
│   ├── data/             # Ingestion logic & validators.
│   ├── features/         # Feature engineering pipeline.
│   ├── models/           # Training and evaluation.
│   ├── serving/          # FastAPI app and predictor wrapper.
│   └── workflows/        # Prefect orchestration.
└── tests/                # Pytest unit tests.
```

## Data Validation & Versioning
- Ingestion validates each record against `src/data/validators.py`; invalid rows are logged and skipped.
- Track data snapshots with DVC:
  ```bash
  dvc init
  dvc remote add -d storage gs://<bucket>/finance-math
  dvc add data/raw data/processed
  git add data/raw.dvc data/processed.dvc .dvc/config
  dvc push
  ```
  Log `.dvc` hashes or `dvc status -c` output to MLflow runs for data-model traceability. Other machines can run `dvc pull` to restore the same snapshot.

## Docker Compose
Run the inference service together with Prometheus and Grafana:
```bash
cd docker
docker compose up --build
```
- Inference: `http://localhost:8000` (metrics at `/metrics`).
- Prometheus: `http://localhost:9090`.
- Grafana: `http://localhost:3000` (default credentials `admin` / `admin`), with the “FinanceMath Inference Overview” dashboard showing request counts, latency (p95), and prediction totals.

## CI/CD
The GitHub Actions workflow (`.github/workflows/mlops-demo.yml`) installs dependencies, runs linting, executes pytest with model loading disabled, and builds the inference Docker image. Extend it with additional jobs (e.g. drift reporting, Prefect deployment triggers, Docker push) as needed.

## Next Steps
- Schedule Prefect deployments or trigger them from GitHub Actions.
- Point MLflow tracking to a shared backend (PostgreSQL, Databricks, etc.).
- Improve model performance via hyperparameter search or alternative algorithms.
- Integrate Prometheus metrics with Grafana alerts or third-party monitoring solutions.
