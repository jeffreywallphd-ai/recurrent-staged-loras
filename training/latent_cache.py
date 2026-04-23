"""Latent-cache integration status.

Latent caching is deferred to future work.
This module intentionally exposes explicit no-op helpers only so old imports
continue to resolve, while the active training path never loads or writes cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


LATENT_CACHE_STATUS = "deferred_future_work"


def maybe_load_latent_cache(cache_dir: str | Path, split: str) -> Any | None:
    """No-op loader while latent cache is fully disabled."""
    _ = (cache_dir, split)
    return None


def maybe_write_latent_cache(cache_dir: str | Path, split: str, payload: Any) -> Path | None:
    """No-op writer while latent cache is fully disabled."""
    _ = (cache_dir, split, payload)
    return None
