"""Lightweight dataset utilities for local baseline training runs.

The initial training path uses deterministic synthetic integer-token sequences,
which keeps tests fast and makes behavior reproducible while preserving a clear
interface for later dataset backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


class IntegerSequenceDataset(Dataset[torch.Tensor]):
    """Deterministic integer-token sequence dataset.

    Each item is a single sequence of token IDs with fixed length.
    """

    def __init__(
        self,
        num_examples: int,
        sequence_length: int,
        vocab_size: int,
        seed: int = 0,
    ) -> None:
        if num_examples < 1:
            raise ValueError("num_examples must be >= 1")
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2 for next-token training")
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")

        self.num_examples = num_examples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_examples:
            raise IndexError(idx)

        # Deterministic pattern per example index and seed.
        base = (idx * self.sequence_length + self.seed) % self.vocab_size
        tokens = (torch.arange(self.sequence_length, dtype=torch.long) + base) % self.vocab_size
        return tokens


def collate_integer_sequences(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Batch integer-token sequences into model-ready tensors."""
    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass(slots=True)
class DatasetBundle:
    """Container for train/eval datasets."""

    train: Dataset[torch.Tensor]
    eval: Dataset[torch.Tensor]


def build_train_eval_datasets(name: str, settings: dict[str, Any], vocab_size: int) -> DatasetBundle:
    """Build train/eval splits for the configured dataset name.

    Supported dataset names:
    - ``placeholder_dataset``
    - ``synthetic_integer_sequences``
    """
    if name not in {"placeholder_dataset", "synthetic_integer_sequences"}:
        raise ValueError(f"Unsupported dataset '{name}'. Expected synthetic local dataset.")

    total_examples = int(settings.get("num_examples", 64))
    eval_fraction = float(settings.get("eval_fraction", 0.25))
    sequence_length = int(settings.get("sequence_length", 12))
    seed = int(settings.get("seed", 0))

    eval_examples = max(1, int(total_examples * eval_fraction))
    train_examples = max(1, total_examples - eval_examples)

    train = IntegerSequenceDataset(
        num_examples=train_examples,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        seed=seed,
    )
    eval_ds = IntegerSequenceDataset(
        num_examples=eval_examples,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        seed=seed + 10_000,
    )
    return DatasetBundle(train=train, eval=eval_ds)


def load_dataset(name: str, split: str) -> Any:
    """Compatibility shim for older smoke-only imports.

    New training code should use :func:`build_train_eval_datasets` directly.
    """
    bundle = build_train_eval_datasets(name=name, settings={}, vocab_size=256)
    if split == "train":
        return bundle.train
    if split in {"eval", "validation", "val"}:
        return bundle.eval
    raise ValueError(f"Unsupported split '{split}'")
