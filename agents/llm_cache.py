"""Disk-backed cache for LLM calls, keyed on a hash of (model, prompt parts). Development
re-runs of identical input must not hit the API -- this session burned real credits
re-running the same job descriptions and census rows across many debugging iterations.

One JSON file per cache key, so entries are trivially inspectable (`cat` a file to see
exactly what a given call returned) rather than opaque rows in a database.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"


def cache_key(model: str, prompt_parts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    for part in prompt_parts:
        h.update(b"\x00")
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def get_cached(model: str, prompt_parts: list[str], cache_dir: Path | None = None) -> dict[str, Any] | None:
    # Resolved at call time (not bound as a default-argument value) so tests can redirect
    # storage by monkeypatching the module-level DEFAULT_CACHE_DIR.
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    path = cache_dir / f"{cache_key(model, prompt_parts)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def set_cached(model: str, prompt_parts: list[str], value: dict[str, Any], cache_dir: Path | None = None) -> None:
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(model, prompt_parts)}.json"
    path.write_text(json.dumps(value, indent=2))
