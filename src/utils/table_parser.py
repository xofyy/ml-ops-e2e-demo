from __future__ import annotations

from typing import Any

import pandas as pd


def markdown_table_to_dataframe(markdown: str) -> pd.DataFrame:
    """Convert a simple GitHub-style markdown table to a pandas DataFrame."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    header_line = lines[0]
    raw_headers = [cell.strip() for cell in header_line.strip("|").split("|")]
    headers = _make_unique_headers(raw_headers)

    data_rows: list[list[str]] = []
    for line in lines[1:]:
        cleaned = line.strip().strip("|")
        stripped = (
            cleaned.replace("-", "").replace(":", "").replace(" ", "").replace("|", "")
        )
        if stripped == "":
            continue
        values = [cell.strip() for cell in cleaned.split("|")]
        if len(values) != len(headers):
            # Pad or truncate rows to align with headers.
            values = (values + [""] * len(headers))[: len(headers)]
        data_rows.append(values)

    return pd.DataFrame(data_rows, columns=headers)


def table_to_records(markdown: str) -> list[dict[str, Any]]:
    """Return table rows as list of dictionaries."""
    df = markdown_table_to_dataframe(markdown)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _make_unique_headers(headers: list[str]) -> list[str]:
    """Ensure column names are unique to avoid pandas warnings."""
    counts: dict[str, int] = {}
    unique_headers: list[str] = []

    for header in headers:
        name = header or "column"
        occurrences = counts.get(name, 0)
        if occurrences == 0:
            unique_headers.append(name)
        else:
            unique_headers.append(f"{name}_{occurrences}")
        counts[name] = occurrences + 1

    return unique_headers
