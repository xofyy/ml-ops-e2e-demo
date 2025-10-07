from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from src.data.ingest import ingest_dataset
from src.features.build_features import run_feature_pipeline
from src.models.evaluate import evaluate_run
from src.models.train import run_training


@task
def ingest_task(config_path: Path | str) -> Path:
    return ingest_dataset(config_path)


@task
def feature_task(data_cfg: Path | str, model_cfg: Path | str) -> Path:
    return run_feature_pipeline(data_cfg, model_cfg)


@task
def train_task(data_cfg: Path | str, model_cfg: Path | str, feature_path: Path) -> str:
    return run_training(data_cfg, model_cfg, feature_path)


@task
def evaluate_task(
    run_id: str, data_cfg: Path | str, model_cfg: Path | str, feature_path: Path
) -> Path:
    return evaluate_run(run_id, data_cfg, model_cfg, feature_path)


@flow(name="finance-math-mlops")
def finance_math_flow(
    data_config_path: Path | str = "configs/dataset.yaml",
    model_config_path: Path | str = "configs/model.yaml",
) -> None:
    ingest_task(data_config_path)
    feature_path = feature_task(data_config_path, model_config_path)
    run_id = train_task(data_config_path, model_config_path, feature_path)
    evaluate_task(run_id, data_config_path, model_config_path, feature_path)


if __name__ == "__main__":
    finance_math_flow()
