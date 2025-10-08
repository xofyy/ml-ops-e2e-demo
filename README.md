# FinanceMath MLOps Demo

FinanceMath MLOps Demo is a lightweight, end-to-end pipeline that ingests the [FinanceMath](https://huggingface.co/datasets/yale-nlp/FinanceMath) dataset, engineers features, trains a regression model, evaluates it, and exposes an inference service. The repository demonstrates reproducible workflows, orchestration, CI/CD, and monitoring practices in an approachable learning setting.

## Features
- **Ingestion**: Download the Hugging Face validation split, normalise markdown tables, validate each record with Pydantic, and persist JSONL artefacts.
- **Feature engineering**: Blend sentence embeddings with numeric table aggregates and metadata features.
- **Training & tracking**: Fit a LightGBM regressor and capture metrics/artefacts via MLflow.
- **Evaluation**: Reuse the feature set to compute MAE/RMSE and store JSON reports.
- **Serving**: Provide a FastAPI service that loads the latest MLflow model.
- **Orchestration**: Prefect flow stitches ingestion -> features -> training -> evaluation.
- **Monitoring**: Evidently script scaffold for drift/performance dashboards.

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

# Update `configs/inference.yaml` to point to your trained model,
#   e.g. set model.model_uri: "runs:/<run_id>/model" or register the model.

# Run the pipeline end to end
python -m src.data.ingest
python -m src.features.build_features
RUN_ID=$(python -m src.models.train)
python -m src.models.evaluate --run-id "$RUN_ID"
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
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

## Repository Layout
```
.
├── configs/              # YAML configs for data, training, inference.
├── data/                 # Data artefacts (consider DVC or Git LFS).
├── docker/               # Dockerfiles for serving.
├── docs/                 # Architecture notes and notebooks.
├── scripts/              # Utility scripts (e.g. Evidently report).
├── src/
│   ├── common/           # Shared config loaders and schemas.
│   ├── data/             # Ingestion logic.
│   ├── features/         # Feature engineering pipeline.
│   ├── models/           # Training and evaluation.
│   ├── serving/          # FastAPI app and predictor wrapper.
│   └── workflows/        # Prefect orchestration.
└── tests/                # Pytest unit tests.
```

## Data Validation & Versioning
- İndirme adımı, her kaydı `src/data/validators.py` altındaki Pydantic şemasıyla doğrular; hatalı kayıtlar log’a yazılıp atlanır.
- Veri versiyonlamak için isteğe bağlı olarak DVC ekleyebilirsin:
  ```bash
  dvc init
  dvc remote add -d storage s3://<bucket>/finance-math
  dvc add data/raw data/processed
  git add data/raw.dvc data/processed.dvc .dvc/config
  ```
  Böylece MLflow run’larında `dvc metrics` veya hash bilgilerini loglayarak veri ile model arasında izlenebilirlik sağlayabilirsin.

## CI/CD
The GitHub Actions workflow (`.github/workflows/mlops-demo.yml`) installs dependencies with pip, runs lint checks, and executes pytest for every push or pull request. Extend it with build/deploy jobs as automation needs grow.

## Next Steps
- Wire Prefect deployments or GitHub Actions to trigger training on merge.
- Connect MLflow to a remote backend (e.g. SQLite, PostgreSQL, Databricks).
- Replace LightGBM with a retrieval + reasoning chain or LLM if desired.
- Expand monitoring to run Evidently on a schedule and push metrics to Grafana.
