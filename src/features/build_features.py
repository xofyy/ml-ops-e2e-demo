from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.common.config import load_config
from src.common.schemas import DatasetSettings, FeatureSettings
from src.data.ingest import load_records
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_numeric(value: object) -> float | None:
    """Attempt to coerce a value to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.replace(",", "").strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def compute_table_features(
    tables: list[list[dict]], aggregations: Iterable[str]
) -> dict[str, float]:
    """Aggregate numeric statistics from parsed table records."""
    numeric_values: list[float] = []
    total_rows = 0

    for table in tables:
        total_rows += len(table)
        for row in table:
            for value in row.values():
                numeric = parse_numeric(value)
                if numeric is not None:
                    numeric_values.append(numeric)

    features: dict[str, float] = {
        "table_count": float(len(tables)),
        "table_row_count": float(total_rows),
        "table_numeric_count": float(len(numeric_values)),
    }

    if numeric_values:
        array = np.array(numeric_values, dtype=float)
        for agg in aggregations:
            if agg == "mean":
                features["table_numeric_mean"] = float(np.mean(array))
            elif agg == "std":
                features["table_numeric_std"] = float(np.std(array))
            elif agg == "max":
                features["table_numeric_max"] = float(np.max(array))
            elif agg == "min":
                features["table_numeric_min"] = float(np.min(array))
            elif agg == "sum":
                features["table_numeric_sum"] = float(np.sum(array))
    else:
        for agg in aggregations:
            features[f"table_numeric_{agg}"] = 0.0

    return features


def build_feature_frame(
    records: list[dict],
    settings: FeatureSettings,
) -> pd.DataFrame:
    LOGGER.info("Loading sentence transformer: %s", settings.features.embedding_model)
    embedder = SentenceTransformer(settings.features.embedding_model)

    df = pd.DataFrame(records)
    question_series = df["question"].fillna("")
    questions = question_series.tolist()
    embeddings = embedder.encode(
        questions, show_progress_bar=False, convert_to_numpy=True
    )
    embed_cols = [f"emb_{idx:03d}" for idx in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embed_cols)

    agg_features = [
        compute_table_features(
            tables=tables, aggregations=settings.features.numeric_aggregations
        )
        for tables in df["tables"]
    ]
    agg_df = pd.DataFrame(agg_features)

    if "topic" in df:
        topic_series = df["topic"].fillna("unknown")
    else:
        topic_series = pd.Series(["unknown"] * len(df), index=df.index, name="topic")
    topic_dummies = pd.get_dummies(topic_series, prefix="topic")

    if "python_solution" in df:
        solution_series = df["python_solution"].fillna("")
    else:
        solution_series = pd.Series(
            [""] * len(df), index=df.index, name="python_solution"
        )
    meta_df = pd.DataFrame(
        {
            "question_length": question_series.str.len().astype(float),
            "solution_length": solution_series.str.len().astype(float),
        }
    )

    feature_frame = pd.concat([embeddings_df, agg_df, topic_dummies, meta_df], axis=1)
    feature_frame["target"] = df["ground_truth"].astype(float)
    feature_frame["question_id"] = df["question_id"]
    feature_frame["topic"] = topic_series
    return feature_frame


def run_feature_pipeline(
    data_config_path: Path | str = "configs/dataset.yaml",
    feature_config_path: Path | str = "configs/model.yaml",
) -> Path:
    """High-level entrypoint: load raw records, build features, persist as parquet."""
    configure_logging()
    data_cfg = load_config(data_config_path, DatasetSettings)
    feature_cfg = load_config(feature_config_path, FeatureSettings)

    raw_path = data_cfg.raw_path()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run ingestion first (src.data.ingest.ingest_dataset)."
        )

    LOGGER.info("Loading records from %s", raw_path)
    records = load_records(raw_path)
    feature_frame = build_feature_frame(records, feature_cfg)

    processed_dir = Path(data_cfg.paths.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "finance_math_features.parquet"
    feature_frame.to_parquet(output_path, index=False)

    LOGGER.info("Saved features to %s", output_path)
    return output_path


if __name__ == "__main__":
    run_feature_pipeline()
