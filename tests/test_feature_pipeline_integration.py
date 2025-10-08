import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.common.schemas import FeatureSettings
from src.models import train as train_module


class _DummySentenceTransformer:
    def __init__(self, model_name: str, **_kwargs):
        self.model_name = model_name

    def encode(
        self,
        sentences: list[str],
        show_progress_bar: bool = False,  # noqa: ARG002
        convert_to_numpy: bool = True,
        batch_size: int | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        base = np.arange(len(sentences) * 4, dtype=float).reshape(len(sentences), 4)
        return base + np.linspace(0.1, 0.4, num=4)


@pytest.mark.integration
def test_feature_frame_with_pca_and_group_split(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        build_features, "SentenceTransformer", _DummySentenceTransformer
    )

    settings = FeatureSettings.model_validate(
        {
            "features": {
                "embedding_model": "dummy-model",
                "include_table_metrics": True,
                "numeric_aggregations": ["mean", "sum"],
                "embedding_cache_dir": None,
                "embedding_batch_size": 2,
                "embedding_dim": 2,
                "feature_version": "test-v1",
            },
            "training": {
                "val_size": 0.25,
                "seed": 7,
                "params": {"n_estimators": 10},
                "n_splits": 2,
                "group_column": "question_id",
            },
            "mlflow": {
                "experiment_name": "test-exp",
                "tracking_uri": "file:./mlruns-test",
            },
        }
    )

    records = [
        {
            "question_id": f"qid-{idx}",
            "question": f"What is value {idx}?",
            "topic": "finance",
            "tables_markdown": [],
            "tables": [[]],
            "python_solution": None,
            "ground_truth": float(idx),
        }
        for idx in range(4)
    ]

    feature_frame, pca_model = build_features.build_feature_frame(records, settings)

    assert isinstance(feature_frame, pd.DataFrame)
    assert "emb_pca_000" in feature_frame.columns
    assert pca_model is not None

    original_frame = feature_frame.copy()
    X_train, X_valid, _, _ = train_module.split_features(
        feature_frame.copy(), settings.training
    )

    train_ids = set(original_frame.loc[X_train.index, "question_id"])
    valid_ids = set(original_frame.loc[X_valid.index, "question_id"])
    assert train_ids.isdisjoint(
        valid_ids
    ), "Group-based split should keep question_ids in one fold"
