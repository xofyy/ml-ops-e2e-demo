from __future__ import annotations

import logging
from pathlib import Path
import hashlib
import subprocess
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GroupKFold

from src.common.config import load_config
from src.common.schemas import DatasetSettings, FeatureSettings
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def load_feature_set(
    dataset_cfg: DatasetSettings, feature_path: Path | None = None
) -> pd.DataFrame:
    """Load the processed parquet features file."""
    if feature_path is None:
        feature_path = (
            Path(dataset_cfg.paths.processed) / "finance_math_features.parquet"
        )
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {feature_path}. Run src.features.build_features.run_feature_pipeline first."
        )
    LOGGER.info("Loading features from %s", feature_path)
    return pd.read_parquet(feature_path)


def split_features(
    feature_frame: pd.DataFrame,
    training_cfg: FeatureSettings.TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = feature_frame.pop("target")
    groups = None
    group_column = training_cfg.group_column
    if group_column and group_column in feature_frame.columns:
        groups = feature_frame[group_column].copy()
    elif group_column:
        LOGGER.warning(
            "Group column '%s' not present in feature frame; falling back to random split.",
            group_column,
        )
    drop_columns = ["question_id", "topic"]
    if group_column:
        drop_columns.append(group_column)
    feature_frame = feature_frame.drop(columns=drop_columns, errors="ignore")

    if groups is not None:
        unique_group_count = pd.Series(groups).nunique()
        if unique_group_count >= training_cfg.n_splits:
            try:
                splitter = GroupKFold(n_splits=training_cfg.n_splits)
                split_iter = splitter.split(feature_frame, target, groups)
                train_idx, valid_idx = next(split_iter)
                return (
                    feature_frame.iloc[train_idx],
                    feature_frame.iloc[valid_idx],
                    target.iloc[train_idx],
                    target.iloc[valid_idx],
                )
            except ValueError as err:
                LOGGER.warning(
                    "GroupKFold failed (%s); falling back to random split.", err
                )
        else:
            LOGGER.warning(
                "Insufficient unique groups (%s) for n_splits=%s; using random split.",
                unique_group_count,
                training_cfg.n_splits,
            )

    X_train, X_valid, y_train, y_valid = train_test_split(
        feature_frame,
        target,
        test_size=training_cfg.val_size,
        random_state=training_cfg.seed,
    )
    return X_train, X_valid, y_train, y_valid


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict,
) -> tuple[LGBMRegressor, dict[str, float]]:
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    metrics = {
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
    }
    return model, metrics


def run_training(
    data_config_path: Path | str = "configs/dataset.yaml",
    feature_config_path: Path | str = "configs/model.yaml",
    feature_path: Path | None = None,
) -> str:
    """End-to-end training entry point."""
    configure_logging()
    data_cfg = load_config(data_config_path, DatasetSettings)
    feature_cfg = load_config(feature_config_path, FeatureSettings)

    feature_frame = load_feature_set(data_cfg, feature_path)

    X_train, X_valid, y_train, y_valid = split_features(
        feature_frame.copy(),
        training_cfg=feature_cfg.training,
    )

    mlflow.set_tracking_uri(feature_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(feature_cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(feature_cfg.training.params)
        mlflow.log_param("val_size", feature_cfg.training.val_size)
        mlflow.log_param("seed", feature_cfg.training.seed)
        mlflow.log_param("feature_version", feature_cfg.features.feature_version)
        mlflow.log_param("feature_count", X_train.shape[1])
        log_dvc_snapshot()
        mlflow.log_dict(
            {
                "feature_columns": list(X_train.columns),
                "version": feature_cfg.features.feature_version,
            },
            artifact_file="feature_columns.json",
        )

        _log_preprocessing_artifacts(data_cfg, feature_cfg)

        model, metrics = train_lightgbm(
            X_train,
            y_train,
            X_valid,
            y_valid,
            feature_cfg.training.params,
        )

        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        LOGGER.info("Validation metrics: %s", metrics)
        return run.info.run_id


if __name__ == "__main__":
    run_training()


def log_dvc_snapshot() -> None:
    """Log DVC status and tracked hashes to the current MLflow run, if available."""
    if not mlflow.active_run():
        return

    status = _safe_run_command(["dvc", "status", "-c"])
    if status:
        mlflow.log_text(status, "dvc_status.txt")

    for dvc_file in ("data/raw.dvc", "data/processed.dvc"):
        digest = _hash_file_if_exists(dvc_file)
        if digest:
            mlflow.log_param(f"dvc_hash_{Path(dvc_file).stem}", digest)


def _safe_run_command(cmd: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()


def _hash_file_if_exists(path_str: str) -> Optional[str]:
    path = Path(path_str)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log_preprocessing_artifacts(
    dataset_cfg: DatasetSettings, feature_cfg: FeatureSettings
) -> None:
    processed_dir = Path(dataset_cfg.paths.processed)
    pca_path = processed_dir / "embedding_pca.joblib"
    if pca_path.exists():
        mlflow.log_artifact(str(pca_path), artifact_path="preprocessing")

    invalid_path = (
        Path(dataset_cfg.paths.raw) / dataset_cfg.ingest.invalid_output_filename
    )
    if invalid_path.exists():
        mlflow.log_artifact(str(invalid_path), artifact_path="data_quality")
