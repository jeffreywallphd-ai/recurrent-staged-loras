"""Minimal latent-cache hook for future speed/ablation experiments.

This module is intentionally lightweight in the current prompt: hooks are
wired in training, but real latent-state serialization is deferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def maybe_load_latent_cache(cache_dir: str | Path, split: str) -> Any | None:
    """Load cached base hidden states if available.

    Current behavior: return ``None`` unless a placeholder marker exists.
    """
    cache_path = Path(cache_dir) / f"{split}.latent_cache.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def maybe_write_latent_cache(cache_dir: str | Path, split: str, payload: Any) -> Path:
    """Persist a placeholder latent-cache payload for future extension."""
    cache_path = Path(cache_dir) / f"{split}.latent_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"deferred": True, "payload": payload}, indent=2), encoding="utf-8")
    return cache_path
