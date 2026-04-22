"""Smoke tests for configuration templates and baseline validation."""

from training.baseline_selector import select_baseline


def test_baseline_selector_accepts_known_baseline() -> None:
    cfg = {"baseline": "shared_recurrence"}
    assert select_baseline(cfg) == "shared_recurrence"
