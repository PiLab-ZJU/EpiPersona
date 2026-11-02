"""Repository-relative paths. Resolution does not depend on the process cwd."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    """Fill unset variables from repo_root/.env. Existing process values win."""
    path = repo_root() / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def resolve_path(env_name: str, default: Path) -> Path:
    """Environment override, else a path already anchored at the repo root."""
    load_dotenv()
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        path = default
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
    return path.resolve()


def data_root() -> Path:
    return resolve_path("EPIPERSONA_DATA_ROOT", repo_root() / "data")


def cache_root() -> Path:
    return resolve_path("EPIPERSONA_CACHE_ROOT", repo_root() / "cache")


def output_root() -> Path:
    return resolve_path("EPIPERSONA_OUTPUT_ROOT", repo_root() / "outputs")


def log_root() -> Path:
    return resolve_path("EPIPERSONA_LOG_ROOT", repo_root() / "logs")


def chat_template_path(filename: str) -> Path:
    path = repo_root() / "chat_templates" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing chat template: {path}")
    return path
