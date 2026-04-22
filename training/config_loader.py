"""Experiment configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment config.

    TODO: add YAML/TOML support and richer schema validation.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
