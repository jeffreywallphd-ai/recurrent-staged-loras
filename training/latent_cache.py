"""Placeholder latent-cache workflow for future speed/ablation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def maybe_load_latent_cache(cache_dir: str | Path, split: str) -> Any | None:
    """Load cached base hidden states if available.

    TODO: define stable serialization format and cache versioning.
    """
    _ = (cache_dir, split)
    return None


def maybe_write_latent_cache(cache_dir: str | Path, split: str, payload: Any) -> Path:
    """Persist latent cache payload for later runs.

    TODO: implement deterministic naming and integrity checks.
    """
    cache_path = Path(cache_dir) / f"{split}.latent_cache.placeholder"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("TODO: latent cache format", encoding="utf-8")
    return cache_path
