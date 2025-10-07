from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

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
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = feature_frame.pop("target")
    feature_frame = feature_frame.drop(
        columns=["question_id", "topic"], errors="ignore"
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        feature_frame,
        target,
        test_size=val_size,
        random_state=seed,
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
        val_size=feature_cfg.training.val_size,
        seed=feature_cfg.training.seed,
    )

    mlflow.set_tracking_uri(feature_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(feature_cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(feature_cfg.training.params)
        mlflow.log_param("val_size", feature_cfg.training.val_size)
        mlflow.log_param("seed", feature_cfg.training.seed)
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_dict(
            {"feature_columns": list(X_train.columns)},
            artifact_file="feature_columns.json",
        )

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
