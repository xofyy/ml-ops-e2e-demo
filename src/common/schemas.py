from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DatasetSettings(BaseModel):
    class DatasetConfig(BaseModel):
        name: str
        split: str
        cache_dir: str | None = None
        use_auth_token: bool | str | None = None

    class PathsConfig(BaseModel):
        raw: str
        processed: str

    class IngestConfig(BaseModel):
        output_filename: str = Field(default="finance_math_validation.jsonl")
        max_examples: int | None = Field(default=None, ge=1)
        invalid_output_filename: str = Field(
            default="finance_math_invalid.jsonl"
        )

    dataset: DatasetConfig
    paths: PathsConfig
    ingest: IngestConfig

    def raw_path(self) -> Path:
        return Path(self.paths.raw) / self.ingest.output_filename


class FeatureSettings(BaseModel):
    class FeatureConfig(BaseModel):
        embedding_model: str
        include_table_metrics: bool = True
        numeric_aggregations: list[Literal["mean", "std", "max", "min", "sum"]] = Field(
            default_factory=lambda: ["mean", "std", "max", "min", "sum"]
        )
        embedding_cache_dir: str | None = None
        embedding_batch_size: int = Field(default=32, ge=1)
        embedding_dim: int | None = Field(default=None, gt=0)
        feature_version: str = "v1"

    class TrainingConfig(BaseModel):
        val_size: float = Field(default=0.2, gt=0, lt=1)
        seed: int = 42
        params: dict[str, float | int | bool | str]
        n_splits: int = Field(default=5, ge=2)
        group_column: str | None = "question_id"

    class MlflowConfig(BaseModel):
        experiment_name: str
        tracking_uri: str

    features: FeatureConfig
    training: TrainingConfig
    mlflow: MlflowConfig


class InferenceSettings(BaseModel):
    class ServiceConfig(BaseModel):
        host: str = "0.0.0.0"
        port: int = Field(default=8000, ge=1, le=65000)

    class ModelConfig(BaseModel):
        registry_uri: str
        model_name: str | None = None
        model_stage: str | None = None
        model_uri: str | None = None

    class LoggingConfig(BaseModel):
        level: str = "INFO"

    service: ServiceConfig
    model: ModelConfig
    logging: LoggingConfig
