from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


class FinanceMathRecord(BaseModel):
    question_id: str
    question: str
    topic: Optional[str] = None
    tables_markdown: list[str]
    tables: list[list[dict[str, Any]]]
    python_solution: Optional[str] = None
    ground_truth: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def validate_tables(self) -> "FinanceMathRecord":
        if not isinstance(self.tables, list):
            raise ValueError("tables must be a list.")
        for table in self.tables:
            if not isinstance(table, list):
                raise ValueError("Each table must be a list of row dictionaries.")
            for row in table:
                if not isinstance(row, dict):
                    raise ValueError("Each table row must be a dictionary.")
        return self


def validate_record(record: dict[str, Any]) -> FinanceMathRecord:
    """Validate a FinanceMath record using Pydantic. Raises ValidationError on failure."""
    return FinanceMathRecord.model_validate(record)


def safe_validate(
    record: dict[str, Any]
) -> tuple[bool, Optional[FinanceMathRecord], Optional[str]]:
    try:
        model = validate_record(record)
        return True, model, None
    except ValidationError as exc:
        return False, None, exc.json()
