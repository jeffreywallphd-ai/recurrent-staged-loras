"""Dataset abstractions for deterministic local baseline training runs.

This module keeps three concerns explicit and separate:
1. Dataset definitions (sequence providers)
2. Sequence generation/loading helpers
3. Batch collation utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import torch
from torch.utils.data import Dataset


Example = dict[str, torch.Tensor]


class SequenceDataset(Dataset[Example]):
    """Simple index-addressable sequence dataset with labels."""

    def __init__(self, examples: list[Example]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")

        sequence_length = int(examples[0]["input_ids"].numel())
        if sequence_length < 2:
            raise ValueError("sequence length must be >= 2")

        for example in examples:
            input_ids = example["input_ids"]
            labels = example["labels"]
            if input_ids.dtype != torch.long or labels.dtype != torch.long:
                raise ValueError("input_ids and labels must be torch.long")
            if int(input_ids.numel()) != sequence_length or int(labels.numel()) != sequence_length:
                raise ValueError("all examples must have equal length")

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        if idx < 0 or idx >= len(self._examples):
            raise IndexError(idx)
        return self._examples[idx]


def _to_next_token_examples(sequences: list[torch.Tensor]) -> list[Example]:
    return [{"input_ids": tokens.clone(), "labels": tokens.clone()} for tokens in sequences]


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

    rng = random.Random(seed)
    sequences: list[torch.Tensor] = []
    for idx in range(num_examples):
        base = (idx * sequence_length + rng.randrange(vocab_size)) % vocab_size
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
    """Generate deterministic, text-like local token patterns."""
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2 for next-token training")
    if vocab_size < 16:
        raise ValueError("vocab_size must be >= 16 for text_style_patterns")

    rng = random.Random(seed)
    templates = [
        [2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13],
        [4, 14, 4, 15, 4, 7],
    ]
    sequences: list[torch.Tensor] = []
    for idx in range(num_examples):
        template = templates[(idx + rng.randrange(len(templates))) % len(templates)]
        values = []
        for pos in range(sequence_length):
            token = template[pos % len(template)]
            token = (token + idx + seed) % vocab_size
            values.append(token)
        sequences.append(torch.tensor(values, dtype=torch.long))
    return sequences


def generate_structured_sequence_examples(
    *,
    num_examples: int,
    prefix_length: int,
    target_length: int,
    vocab_size: int,
    seed: int,
) -> list[Example]:
    """Generate examples with explicit prefix/target structure and token labels."""
    if prefix_length < 1 or target_length < 1:
        raise ValueError("prefix_length and target_length must be >= 1")
    rng = random.Random(seed)
    examples: list[Example] = []
    total_length = prefix_length + target_length
    for idx in range(num_examples):
        prefix_base = rng.randrange(vocab_size)
        prefix = [(prefix_base + step + idx) % vocab_size for step in range(prefix_length)]
        target = [((prefix[-1] if prefix else 0) + step + 1) % vocab_size for step in range(target_length)]
        tokens = torch.tensor(prefix + target, dtype=torch.long)
        if int(tokens.numel()) != total_length:
            raise RuntimeError("invalid structured example length")
        examples.append({"input_ids": tokens, "labels": tokens.clone()})
    return examples


def collate_token_sequences(batch: list[Example]) -> dict[str, torch.Tensor]:
    """Batch integer-token sequences into model-ready tensors."""
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


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
    - ``structured_sequence``
    """
    if name not in {"placeholder_dataset", "synthetic_integer_sequences", "text_style_patterns", "structured_sequence"}:
        raise ValueError(f"Unsupported dataset '{name}'.")

    total_examples = int(settings.get("num_examples", 64))
    eval_fraction = float(settings.get("eval_fraction", 0.25))
    sequence_length = int(settings.get("sequence_length", 12))
    seed = int(settings.get("seed", 0))

    train_examples, eval_examples = _split_counts(total_examples, eval_fraction)

    if name == "structured_sequence":
        prefix_length = int(settings.get("prefix_length", max(1, sequence_length // 2)))
        target_length = int(settings.get("target_length", max(1, sequence_length - prefix_length)))
        train_data = generate_structured_sequence_examples(
            num_examples=train_examples,
            prefix_length=prefix_length,
            target_length=target_length,
            vocab_size=vocab_size,
            seed=seed,
        )
        eval_data = generate_structured_sequence_examples(
            num_examples=eval_examples,
            prefix_length=prefix_length,
            target_length=target_length,
            vocab_size=vocab_size,
            seed=seed + 10_000,
        )
        return DatasetBundle(train=SequenceDataset(train_data), eval=SequenceDataset(eval_data))

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

    return DatasetBundle(
        train=SequenceDataset(_to_next_token_examples(train_sequences)),
        eval=SequenceDataset(_to_next_token_examples(eval_sequences)),
    )


def load_dataset(name: str, split: str) -> Any:
    """Compatibility shim for older smoke-only imports."""
    bundle = build_train_eval_datasets(name=name, settings={}, vocab_size=256)
    if split == "train":
        return bundle.train
    if split in {"eval", "validation", "val"}:
        return bundle.eval
    raise ValueError(f"Unsupported split '{split}'")
