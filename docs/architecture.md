# Architecture Overview

The FinanceMath MLOps demo implements an end-to-end learning pipeline with the following stages:

1. **Data ingestion**
   - Source: Hugging Face dataset `yale-nlp/FinanceMath` (validation split).
   - Output: JSONL file with parsed markdown tables and metadata.
   - Authentication: supports gated datasets via `use_auth_token` (config) and `huggingface-cli login`.
   - Validation: each record is validated against a Pydantic schema; invalid rows are logged and skipped.
2. **Feature engineering**
   - Text embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
   - Numeric table aggregations (mean, std, min, max, sum) and metadata features.
   - Output: `finance_math_features.parquet`.
3. **Model training**
   - Algorithm: LightGBM regressor.
   - Tracking: MLflow local file store (`mlruns`) with metrics and artefacts (`feature_columns.json`).
4. **Model evaluation**
   - Metrics: MAE, RMSE against ground truth from FinanceMath.
   - Artefact: JSON metrics report saved under `data/processed`.
5. **Serving**
   - FastAPI app loads either a registered model stage or a run-specific URI from MLflow.
   - Predictor regenerates feature vector using the same embedding/model config.
   - Prometheus counters/histograms capture request and latency metrics; `/metrics` endpoint exposes them.
6. **Orchestration**
   - Prefect flow orchestrates ingestion -> features -> training -> evaluation.
   - Command-line entrypoints exposed through `python -m`.
7. **Monitoring**
   - Evidently script scaffolding to compute drift and regression performance reports.
   - Extendable to Prometheus/Grafana metrics exporters.

## Deployment Options
- **Local development**: `uvicorn src.serving.app:app` with MLflow file store.
- **Docker compose**: `docker/docker-compose.yml` spins up the inference API and Prometheus for local monitoring.
- **Containerised inference**: Build `docker/inference.Dockerfile` and deploy to Docker/Kubernetes.
- **CI/CD**: GitHub Actions workflow ensures linting and tests on each commit.
- **Scheduling**: Convert Prefect flow into a deployment for periodic re-training.

## Data Management
- Check large artefacts into an external storage solution (DVC, S3, MinIO).
- Cache Hugging Face downloads under `./.cache/hf`.
- Use `mlruns/` as local tracking backend; swap for a remote DB for collaboration.
