from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import artifacts
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.common.config import load_config
from src.common.schemas import FeatureSettings, InferenceSettings
from src.features.build_features import compute_table_features
from src.utils.logging import configure_logging
from src.utils.table_parser import table_to_records

LOGGER = logging.getLogger(__name__)


class FinanceMathRequest(BaseModel):
    question: str
    tables_markdown: list[str] = []
    topic: str | None = None


class Predictor:
    def __init__(
        self,
        inference_cfg: InferenceSettings,
        feature_cfg: FeatureSettings,
    ) -> None:
        self.inference_cfg = inference_cfg
        self.feature_cfg = feature_cfg

        mlflow.set_tracking_uri(inference_cfg.model.registry_uri)
        model_cfg = inference_cfg.model

        if model_cfg.model_uri:
            model_uri = model_cfg.model_uri
        else:
            if not model_cfg.model_name or not model_cfg.model_stage:
                raise ValueError(
                    "Either provide model.model_uri or both model.model_name and model.model_stage."
                )
            model_uri = f"models:/{model_cfg.model_name}/{model_cfg.model_stage}"

        LOGGER.info("Loading model from %s", model_uri)
        self.model = mlflow.pyfunc.load_model(model_uri)

        feature_columns_uri = self._resolve_feature_columns_uri(model_uri)
        metadata = artifacts.load_dict(feature_columns_uri)
        self.feature_columns: list[str] = metadata["feature_columns"]

        self.embedder = SentenceTransformer(feature_cfg.features.embedding_model)

    def build_feature_vector(self, request: FinanceMathRequest) -> pd.DataFrame:
        tables_structured = [table_to_records(tbl) for tbl in request.tables_markdown]
        agg_features = compute_table_features(
            tables_structured, self.feature_cfg.features.numeric_aggregations
        )

        embedding = self.embedder.encode([request.question], show_progress_bar=False)[0]
        feature_dict: dict[str, Any] = {
            f"emb_{idx:03d}": float(value) for idx, value in enumerate(embedding)
        }
        feature_dict.update(agg_features)

        topic = request.topic or "unknown"
        feature_dict["question_length"] = float(len(request.question))
        feature_dict["solution_length"] = 0.0

        for column in self.feature_columns:
            if column.startswith("topic_"):
                feature_dict[column] = 1.0 if column == f"topic_{topic}" else 0.0

        frame = pd.DataFrame([feature_dict])
        return frame.reindex(columns=self.feature_columns, fill_value=0.0)

    def predict(self, request: FinanceMathRequest) -> float:
        feature_vector = self.build_feature_vector(request)
        prediction = self.model.predict(feature_vector)[0]
        return float(prediction)

    def _resolve_feature_columns_uri(self, model_uri: str) -> str:
        if model_uri.startswith("runs:/"):
            rest = model_uri.split("runs:/", 1)[1]
            run_id = rest.split("/", 1)[0]
            return f"runs:/{run_id}/feature_columns.json"
        return f"{model_uri}/feature_columns.json"


def load_predictor(
    inference_config_path: Path | str = "configs/inference.yaml",
    feature_config_path: Path | str = "configs/model.yaml",
) -> Predictor:
    configure_logging()
    inference_cfg = load_config(inference_config_path, InferenceSettings)
    feature_cfg = load_config(feature_config_path, FeatureSettings)
    return Predictor(inference_cfg, feature_cfg)
