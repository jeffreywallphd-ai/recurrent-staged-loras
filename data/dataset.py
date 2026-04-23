"""Real dataset loading + explicit 3-stage preprocessing and collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random
import re

import torch
from torch.utils.data import Dataset


Example = dict[str, torch.Tensor]


class SequenceDataset(Dataset[Example]):
    def __init__(self, examples: list[Example]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


def _extract_reasoning_and_answer(response: str) -> tuple[str, str]:
    if "####" in response:
        reasoning, answer = response.rsplit("####", 1)
        return reasoning.strip(), answer.strip()
    lower = response.lower()
    marker = "the answer is"
    if marker in lower:
        idx = lower.rfind(marker)
        return response[:idx].strip(), response[idx:].strip()
    return response.strip(), ""


def _build_staged_text(problem: str, reasoning: str, final_answer: str) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]:
    s1 = f"Problem:\n{problem.strip()}\n\n"
    s2 = f"Reasoning:\n{reasoning.strip()}\n\n" if reasoning else "Reasoning:\n\n"
    s3 = f"Final Answer:\n{final_answer.strip()}" if final_answer else "Final Answer:\n"
    full = s1 + s2 + s3
    return full, (0, len(s1)), (len(s1), len(s1) + len(s2)), (len(s1) + len(s2), len(full))


def _char_spans_to_token_mask(offset_mapping: list[tuple[int, int]], span: tuple[int, int]) -> torch.Tensor:
    s, e = span
    mask = []
    for tok_s, tok_e in offset_mapping:
        active = tok_e > s and tok_s < e
        mask.append(active)
    return torch.tensor(mask, dtype=torch.bool)


def build_staged_examples_from_hf(
    *,
    subset_size: int,
    seed: int,
    max_seq_length: int,
    tokenizer: Any,
    cache_dir: str | None,
    split: str,
) -> list[Example]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("meta-math/MetaMathQA", split=split, cache_dir=cache_dir)
    order = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(order)
    selected = order[: subset_size]
    examples: list[Example] = []

    for idx in selected:
        row = ds[int(idx)]
        problem = str(row.get("query", "")).strip()
        response = str(row.get("response", "")).strip()
        if not problem or not response:
            continue
        reasoning, answer = _extract_reasoning_and_answer(response)
        staged_text, stage1_span, stage2_span, stage3_span = _build_staged_text(problem, reasoning, answer)

        tok = tokenizer(
            staged_text,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        input_ids = torch.tensor(tok["input_ids"], dtype=torch.long)
        offsets = [(int(a), int(b)) for a, b in tok["offset_mapping"]]
        stage1_mask = _char_spans_to_token_mask(offsets, stage1_span)
        stage2_mask = _char_spans_to_token_mask(offsets, stage2_span)
        stage3_mask = _char_spans_to_token_mask(offsets, stage3_span)

        # force disjoint preference to later stage when overlaps occur
        stage2_mask = stage2_mask & ~stage3_mask
        stage1_mask = stage1_mask & ~stage2_mask & ~stage3_mask

        labels = input_ids.clone()
        if int(stage3_mask.sum().item()) == 0:
            continue

        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "stage1_mask": stage1_mask,
                "stage2_mask": stage2_mask,
                "stage3_mask": stage3_mask,
                "has_stage2": torch.tensor(int(stage2_mask.sum().item()) > 0, dtype=torch.bool),
                "numeric_answer": torch.tensor(bool(re.search(r"[-+]?\d*\.?\d+", answer)), dtype=torch.bool),
            }
        )
    if not examples:
        raise RuntimeError("No valid examples were built from MetaMathQA.")
    return examples


def build_test_examples(*, num_examples: int, sequence_length: int, vocab_size: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for i in range(num_examples):
        base = rng.randrange(vocab_size)
        ids = (torch.arange(sequence_length, dtype=torch.long) + base + i) % vocab_size
        thirds = sequence_length // 3
        s1 = torch.zeros(sequence_length, dtype=torch.bool)
        s2 = torch.zeros(sequence_length, dtype=torch.bool)
        s3 = torch.zeros(sequence_length, dtype=torch.bool)
        s1[:thirds] = True
        s2[thirds : 2 * thirds] = True
        s3[2 * thirds :] = True
        out.append({"input_ids": ids, "labels": ids.clone(), "stage1_mask": s1, "stage2_mask": s2, "stage3_mask": s3})
    return out


def collate_token_sequences(batch: list[Example]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].shape[0]) for item in batch)

    def _pad_1d(x: torch.Tensor, pad_val: int = 0) -> torch.Tensor:
        if x.shape[0] == max_len:
            return x
        pad = torch.full((max_len - x.shape[0],), pad_val, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([_pad_1d(item["input_ids"], 0) for item in batch])
    labels = torch.stack([_pad_1d(item["labels"], -100) for item in batch])
    attention_mask = (input_ids != 0).long()

    collated = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    for key in ("stage1_mask", "stage2_mask", "stage3_mask"):
        collated[key] = torch.stack([_pad_1d(item[key].bool(), 0).bool() for item in batch])
    return collated


@dataclass(slots=True)
class DatasetBundle:
    train: SequenceDataset
    eval: SequenceDataset


def _split_counts(total_examples: int, eval_fraction: float) -> tuple[int, int]:
    eval_examples = max(1, int(total_examples * eval_fraction))
    train_examples = max(1, total_examples - eval_examples)
    return train_examples, eval_examples


def build_train_eval_datasets(name: str, settings: dict[str, Any], vocab_size: int, tokenizer: Any | None = None) -> DatasetBundle:
    del vocab_size
    total_examples = int(settings.get("subset_size", 25_000))
    eval_fraction = float(settings.get("eval_fraction", 0.1))
    seed = int(settings.get("seed", 0))
    train_count, eval_count = _split_counts(total_examples, eval_fraction)

    if name == "test_synthetic_stage_dataset":
        seq_len = int(settings.get("sequence_length", 12))
        train = build_test_examples(num_examples=train_count, sequence_length=seq_len, vocab_size=512, seed=seed)
        evalv = build_test_examples(num_examples=eval_count, sequence_length=seq_len, vocab_size=512, seed=seed + 1)
        return DatasetBundle(train=SequenceDataset(train), eval=SequenceDataset(evalv))

    if name != "metamath_qa":
        raise ValueError(f"Unsupported dataset '{name}'.")
    if tokenizer is None:
        raise ValueError("Tokenizer is required for metamath_qa dataset.")

    examples = build_staged_examples_from_hf(
        subset_size=total_examples,
        seed=seed,
        max_seq_length=int(settings.get("max_seq_length", 2048)),
        tokenizer=tokenizer,
        cache_dir=settings.get("cache_dir"),
        split=str(settings.get("split", "train")),
    )
    return DatasetBundle(train=SequenceDataset(examples[:train_count]), eval=SequenceDataset(examples[train_count : train_count + eval_count]))
