"""Dataset abstractions for deterministic local baseline training runs.

This module keeps three concerns explicit and separate:
1. Dataset definitions (sequence providers)
2. Sequence generation/loading helpers
3. Batch collation utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset[torch.Tensor]):
    """Simple index-addressable sequence dataset."""

    def __init__(self, sequences: list[torch.Tensor]) -> None:
        if not sequences:
            raise ValueError("sequences must not be empty")
        sequence_length = int(sequences[0].numel())
        if sequence_length < 2:
            raise ValueError("sequence length must be >= 2")

        for sequence in sequences:
            if sequence.dtype != torch.long:
                raise ValueError("all sequences must be torch.long")
            if int(sequence.numel()) != sequence_length:
                raise ValueError("all sequences must have equal length")

        self._sequences = sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self._sequences):
            raise IndexError(idx)
        return self._sequences[idx]


def generate_synthetic_integer_sequences(
    *,
    num_examples: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    """Generate deterministic integer-token sequences."""
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2 for next-token training")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")

    sequences: list[torch.Tensor] = []
    for idx in range(num_examples):
        base = (idx * sequence_length + seed) % vocab_size
        tokens = (torch.arange(sequence_length, dtype=torch.long) + base) % vocab_size
        sequences.append(tokens)
    return sequences


def generate_text_style_sequences(
    *,
    num_examples: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    """Generate deterministic, text-like local token patterns.

    The patterns emulate repeated phrase structures and delimiter-like tokens.
    """
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2 for next-token training")
    if vocab_size < 16:
        raise ValueError("vocab_size must be >= 16 for text_style_patterns")

    templates = [
        [2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13],
        [4, 14, 4, 15, 4, 7],
    ]
    sequences: list[torch.Tensor] = []
    for idx in range(num_examples):
        template = templates[(idx + seed) % len(templates)]
        values = []
        for pos in range(sequence_length):
            token = template[pos % len(template)]
            token = (token + idx + seed) % vocab_size
            values.append(token)
        sequences.append(torch.tensor(values, dtype=torch.long))
    return sequences


def collate_token_sequences(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Batch integer-token sequences into model-ready tensors."""
    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass(slots=True)
class DatasetBundle:
    """Container for train/eval datasets."""

    train: SequenceDataset
    eval: SequenceDataset


def _split_counts(total_examples: int, eval_fraction: float) -> tuple[int, int]:
    eval_examples = max(1, int(total_examples * eval_fraction))
    train_examples = max(1, total_examples - eval_examples)
    return train_examples, eval_examples


def build_train_eval_datasets(name: str, settings: dict[str, Any], vocab_size: int) -> DatasetBundle:
    """Build train/eval splits for supported local dataset modes.

    Supported dataset names:
    - ``placeholder_dataset`` (alias of ``synthetic_integer_sequences``)
    - ``synthetic_integer_sequences``
    - ``text_style_patterns``
    """
    if name not in {"placeholder_dataset", "synthetic_integer_sequences", "text_style_patterns"}:
        raise ValueError(f"Unsupported dataset '{name}'.")

    total_examples = int(settings.get("num_examples", 64))
    eval_fraction = float(settings.get("eval_fraction", 0.25))
    sequence_length = int(settings.get("sequence_length", 12))
    seed = int(settings.get("seed", 0))

    train_examples, eval_examples = _split_counts(total_examples, eval_fraction)

    generator = generate_synthetic_integer_sequences
    if name == "text_style_patterns":
        generator = generate_text_style_sequences

    train_sequences = generator(
        num_examples=train_examples,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        seed=seed,
    )
    eval_sequences = generator(
        num_examples=eval_examples,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        seed=seed + 10_000,
    )

    return DatasetBundle(train=SequenceDataset(train_sequences), eval=SequenceDataset(eval_sequences))


def load_dataset(name: str, split: str) -> Any:
    """Compatibility shim for older smoke-only imports."""
    bundle = build_train_eval_datasets(name=name, settings={}, vocab_size=256)
    if split == "train":
        return bundle.train
    if split in {"eval", "validation", "val"}:
        return bundle.eval
    raise ValueError(f"Unsupported split '{split}'")
