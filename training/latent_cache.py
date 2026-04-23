"""Latent-cache integration status.

Latent-cache serialization is intentionally disabled for this prompt stage.
The helpers below are explicit no-ops to avoid ambiguity in behavior while
keeping a stable hook point for future work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


LATENT_CACHE_STATUS = "disabled_future_work"


def maybe_load_latent_cache(cache_dir: str | Path, split: str) -> Any | None:
    """No-op loader while latent cache is disabled."""
    _ = (cache_dir, split)
    return None


def maybe_write_latent_cache(cache_dir: str | Path, split: str, payload: Any) -> Path | None:
    """No-op writer while latent cache is disabled."""
    _ = (cache_dir, split, payload)
    return None
