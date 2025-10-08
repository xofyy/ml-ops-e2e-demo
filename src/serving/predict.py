from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import mlflow
import pandas as pd
from mlflow import artifacts
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib

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

        self.model_uri = model_uri
        LOGGER.info("Loading model from %s", model_uri)
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.run_id: Optional[str] = getattr(self.model.metadata, "run_id", None)

        feature_columns_uri = self._resolve_feature_columns_uri()
        metadata = artifacts.load_dict(feature_columns_uri)
        self.feature_columns: list[str] = metadata["feature_columns"]
        self.feature_version: Optional[str] = metadata.get("version")
        self.embedding_columns = [
            column for column in self.feature_columns if column.startswith("emb")
        ]

        self.embedding_reducer = self._load_embedding_reducer()

        embedder_kwargs: dict[str, object] = {}
        cache_dir = feature_cfg.features.embedding_cache_dir
        if cache_dir:
            embedder_kwargs["cache_folder"] = cache_dir
        self.embedder = SentenceTransformer(
            feature_cfg.features.embedding_model, **embedder_kwargs
        )


    def build_feature_vector(self, request: FinanceMathRequest) -> pd.DataFrame:
        tables_structured = [table_to_records(tbl) for tbl in request.tables_markdown]
        agg_features = compute_table_features(
            tables_structured, self.feature_cfg.features.numeric_aggregations
        )

        embedding_array = self.embedder.encode(
            [request.question], show_progress_bar=False
        )[0]
        if self.embedding_reducer is not None:
            embedding_array = self.embedding_reducer.transform(
                [embedding_array]
            )[0]
        feature_dict: dict[str, Any] = {}
        for idx, column in enumerate(self.embedding_columns):
            if idx < len(embedding_array):
                feature_dict[column] = float(embedding_array[idx])
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

    def _resolve_feature_columns_uri(self) -> str:
        run_id = self.run_id
        if run_id:
            return f"runs:/{run_id}/feature_columns.json"
        return f"{self.model_uri}/feature_columns.json"

    def _resolve_pca_uri(self) -> str:
        run_id = self.run_id
        if run_id:
            return f"runs:/{run_id}/preprocessing/embedding_pca.joblib"
        return str(Path(self.model_uri).parent / "preprocessing" / "embedding_pca.joblib")

    def _load_embedding_reducer(self):
        pca_uri = self._resolve_pca_uri()
        try:
            if pca_uri.startswith(("runs:/", "models:/")):
                local_path = artifacts.load_artifact(pca_uri)
            else:
                local_path = Path(pca_uri)
                if not local_path.exists():
                    raise FileNotFoundError(pca_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.info("No PCA transformer found at %s (%s)", pca_uri, exc)
            return None
        payload = joblib.load(str(local_path))
        if isinstance(payload, dict) and "transformer" in payload:
            return payload["transformer"]
        return payload


def load_predictor(
    inference_config_path: Path | str = "configs/inference.yaml",
    feature_config_path: Path | str = "configs/model.yaml",
) -> Predictor:
    configure_logging()
    inference_cfg = load_config(inference_config_path, InferenceSettings)
    feature_cfg = load_config(feature_config_path, FeatureSettings)
    return Predictor(inference_cfg, feature_cfg)
