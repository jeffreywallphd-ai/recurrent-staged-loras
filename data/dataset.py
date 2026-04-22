"""Dataset loading placeholders for training/evaluation splits."""

from __future__ import annotations

from typing import Any


def load_dataset(name: str, split: str) -> Any:
    """Load a dataset split by name.

    TODO: implement dataset registry and preprocessing hooks.
    """
    raise NotImplementedError("TODO: implement dataset loading")
