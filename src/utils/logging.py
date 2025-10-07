from __future__ import annotations

import logging


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logger with reasonable defaults."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
