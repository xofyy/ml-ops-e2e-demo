from __future__ import annotations

import json
import logging
from pathlib import Path
from datasets import load_dataset

from src.common.config import load_config
from src.common.schemas import DatasetSettings
from src.utils.logging import configure_logging
from src.utils.table_parser import table_to_records

LOGGER = logging.getLogger(__name__)


def prepare_record(example: dict) -> dict:
    """Transform a FinanceMath raw example into a JSON-serialisable record."""
    tables_markdown: list[str] = example.get("tables", [])
    return {
        "question_id": example["question_id"],
        "question": example["question"],
        "topic": example.get("topic"),
        "tables_markdown": tables_markdown,
        "tables": [table_to_records(markdown) for markdown in tables_markdown],
        "python_solution": example.get("python_solution"),
        "ground_truth": example.get("ground_truth"),
    }


def ingest_dataset(config_path: Path | str = "configs/dataset.yaml") -> Path:
    """Fetch the FinanceMath dataset and persist it to a JSONL file."""
    configure_logging()
    settings = load_config(config_path, DatasetSettings)

    dataset_cfg = settings.dataset
    LOGGER.info("Loading dataset %s split=%s", dataset_cfg.name, dataset_cfg.split)
    load_kwargs: dict[str, object] = {
        "path": dataset_cfg.name,
        "split": dataset_cfg.split,
    }
    if dataset_cfg.cache_dir:
        load_kwargs["cache_dir"] = dataset_cfg.cache_dir
    if dataset_cfg.use_auth_token is not None:
        load_kwargs["token"] = dataset_cfg.use_auth_token

    hf_dataset = load_dataset(**load_kwargs)

    raw_dir = Path(settings.paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / settings.ingest.output_filename

    limit = settings.ingest.max_examples
    LOGGER.info("Writing records to %s (limit=%s)", output_path, limit or "all")

    written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(hf_dataset):  # type: ignore[arg-type]
            if limit is not None and index >= limit:
                break
            record = prepare_record(example)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    LOGGER.info("Ingestion complete. Total records written: %s", written)
    return output_path


def load_records(path: Path | str) -> list[dict]:
    """Convenience helper to load JSONL records back into memory."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    ingest_dataset()
