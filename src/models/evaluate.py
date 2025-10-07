from __future__ import annotations

import json
import logging
from pathlib import Path
import argparse

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.config import load_config
from src.common.schemas import DatasetSettings, FeatureSettings
from src.models.train import load_feature_set
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def evaluate_run(
    run_id: str,
    data_config_path: Path | str = "configs/dataset.yaml",
    feature_config_path: Path | str = "configs/model.yaml",
    feature_path: Path | None = None,
) -> Path:
    """Evaluate a trained model run against the full feature dataset."""
    configure_logging()
    data_cfg = load_config(data_config_path, DatasetSettings)
    feature_cfg = load_config(feature_config_path, FeatureSettings)
    feature_frame = load_feature_set(data_cfg, feature_path)

    if "target" not in feature_frame:
        raise KeyError("Feature frame must contain 'target' column for evaluation.")

    X = feature_frame.drop(columns=["target", "question_id", "topic"], errors="ignore")
    y_true = feature_frame["target"]

    mlflow.set_tracking_uri(feature_cfg.mlflow.tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    LOGGER.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    predictions = model.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }

    LOGGER.info("Evaluation metrics: %s", metrics)

    processed_dir = Path(data_cfg.paths.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path = processed_dir / "evaluation_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an MLflow run on FinanceMath features."
    )
    parser.add_argument(
        "--run-id", required=True, help="MLflow run identifier to evaluate."
    )
    parser.add_argument("--data-config", default="configs/dataset.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--feature-path", default=None)
    args = parser.parse_args()

    feature_path = Path(args.feature_path) if args.feature_path else None
    evaluate_run(
        run_id=args.run_id,
        data_config_path=args.data_config,
        feature_config_path=args.model_config,
        feature_path=feature_path,
    )
