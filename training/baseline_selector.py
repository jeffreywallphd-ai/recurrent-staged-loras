"""Baseline selection logic for controlled variant instantiation."""

from __future__ import annotations

from typing import Any


VALID_BASELINES = {
    "base",
    "standard_lora",
    "latent_refiner_only",
    "shared_recurrence",
    "stage_specialized_recurrence",
}


def select_baseline(config: dict[str, Any]) -> str:
    """Resolve and validate baseline name from a loaded experiment config."""
    baseline = config.get("baseline")
    if baseline not in VALID_BASELINES:
        raise ValueError(f"Unknown baseline '{baseline}'. Expected one of {sorted(VALID_BASELINES)}")
    return baseline
