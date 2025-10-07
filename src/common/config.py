from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a standard Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path | str, model: type[ModelT]) -> ModelT:
    """Load a YAML configuration file and validate it against a pydantic model."""
    data = load_yaml(path)
    return model.model_validate(data)
