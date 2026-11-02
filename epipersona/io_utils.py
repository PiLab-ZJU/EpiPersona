"""Small JSON helpers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False)


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_jsonl(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def extract_dict(text: str):
    """Return the first JSON-like object in text, or None."""
    if not text:
        return None
    normalized = (
        text.replace("null", "None")
        .replace(": true", ": True")
        .replace(":true", ": True")
        .replace(": false", ": False")
        .replace(":false", ": False")
    )
    match = re.search(r"\{(.*)\}", normalized, re.DOTALL)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group())
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None
