"""Dataset builders for staged training/evaluation with reproducibility identity.

Pipeline role:
- load raw sources (MetaMathQA + optional external datasets),
- convert rows into staged token sequences and answer spans,
- emit deterministic identity fingerprints/hashes used for run pairing checks.

Invariant: confirmatory comparisons rely on dataset fingerprints and eval-sample
hashes generated here; changing identity logic changes statistical comparability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import random
import re

import torch
from torch.utils.data import Dataset

from training.answer_eval import extract_numeric_values, normalize_answer_text

Example = dict[str, torch.Tensor | str]


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
    boxed = re.findall(r"\\boxed\{([^}]*)\}", response)
    if boxed:
        answer = boxed[-1].strip()
        anchor = response.rfind("\\boxed")
        reasoning = response[:anchor].strip() if anchor >= 0 else response.strip()
        return reasoning, answer
    lower = response.lower()
    marker = "the answer is"
    if marker in lower:
        idx = lower.rfind(marker)
        trailing = response[idx + len(marker) :].strip(" :.\n\t")
        return response[:idx].strip(), trailing
    return response.strip(), ""


def _build_staged_text(
    problem: str, reasoning: str, final_answer: str
) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    s1 = f"Problem:\n{problem.strip()}\n\n"
    s2 = f"Reasoning:\n{reasoning.strip()}\n\n" if reasoning else "Reasoning:\n\n"
    final_answer_prefix = "Final Answer:\n"
    s3 = f"{final_answer_prefix}{final_answer.strip()}" if final_answer else final_answer_prefix
    full = s1 + s2 + s3
    stage1_span = (0, len(s1))
    stage2_span = (len(s1), len(s1) + len(s2))
    stage3_span = (len(s1) + len(s2), len(full))
    answer_span_start = stage3_span[0] + len(final_answer_prefix)
    answer_span = (answer_span_start, len(full))
    return full, stage1_span, stage2_span, stage3_span, answer_span


def _build_plain_text(problem: str, answer: str) -> tuple[str, tuple[int, int]]:
    prompt = f"Question:\n{problem.strip()}\n\nAnswer:\n"
    target = answer.strip()
    full = f"{prompt}{target}"
    return full, (len(prompt), len(full))


def _char_spans_to_token_mask(offset_mapping: list[tuple[int, int]], span: tuple[int, int]) -> torch.Tensor:
    s, e = span
    mask = []
    for tok_s, tok_e in offset_mapping:
        active = tok_e > s and tok_s < e
        mask.append(active)
    return torch.tensor(mask, dtype=torch.bool)


def _example_from_text(
    *,
    tokenizer: Any,
    text: str,
    answer_span: tuple[int, int],
    max_seq_length: int,
    answer_text: str,
    source_signature: str,
) -> tuple[Example | None, bool, bool]:
    """Tokenize one QA text and derive answer-span masks.

    Returns:
        (example_or_none, truncated_anywhere, answer_span_truncated)
    """
    tok = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = torch.tensor(tok["input_ids"], dtype=torch.long)
    offsets = [(int(a), int(b)) for a, b in tok["offset_mapping"]]
    answer_mask = _char_spans_to_token_mask(offsets, answer_span)
    stage3_mask = torch.ones_like(answer_mask, dtype=torch.bool)

    max_char_covered = max((b for _a, b in offsets), default=0)
    truncated = max_char_covered < len(text)
    answer_truncated = answer_span[1] > max_char_covered
    if int(answer_mask.sum().item()) == 0:
        return None, truncated, answer_truncated

    labels = input_ids.clone()
    final_answer_text = answer_text.strip()
    ex: Example = {
        "input_ids": input_ids,
        "labels": labels,
        "stage1_mask": torch.zeros_like(answer_mask, dtype=torch.bool),
        "stage2_mask": torch.zeros_like(answer_mask, dtype=torch.bool),
        "stage3_mask": stage3_mask,
        "answer_mask": answer_mask,
        "final_answer_mask": answer_mask,
        "answer_text": final_answer_text,
        "answer_text_normalized": normalize_answer_text(final_answer_text),
        "source_signature": source_signature,
    }
    return ex, truncated, answer_truncated


def build_staged_examples_from_hf(
    *,
    subset_size: int,
    seed: int,
    max_seq_length: int,
    tokenizer: Any,
    cache_dir: str | None,
    split: str,
) -> tuple[list[Example], dict[str, int]]:
    """Build staged MetaMathQA examples with filtering diagnostics."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("meta-math/MetaMathQA", split=split, cache_dir=cache_dir)
    order = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(order)
    selected = order[: subset_size]
    examples: list[Example] = []

    raw_selected = len(selected)
    filtered_empty = 0
    filtered_missing_stage3 = 0
    stage2_non_empty = 0
    stage3_non_empty = 0
    valid_answer_spans = 0
    numeric_answers = 0
    truncated_examples = 0
    answer_span_truncated = 0

    for idx in selected:
        row = ds[int(idx)]
        problem = str(row.get("query", "")).strip()
        response = str(row.get("response", "")).strip()
        if not problem or not response:
            filtered_empty += 1
            continue

        reasoning, answer = _extract_reasoning_and_answer(response)
        staged_text, stage1_span, stage2_span, stage3_span, answer_span = _build_staged_text(problem, reasoning, answer)

        tok = tokenizer(
            staged_text,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        input_ids = torch.tensor(tok["input_ids"], dtype=torch.long)
        offsets = [(int(a), int(b)) for a, b in tok["offset_mapping"]]
        max_char_covered = max((b for _a, b in offsets), default=0)
        if max_char_covered < len(staged_text):
            truncated_examples += 1
            if answer_span[1] > max_char_covered:
                answer_span_truncated += 1
        stage1_mask = _char_spans_to_token_mask(offsets, stage1_span)
        stage2_mask = _char_spans_to_token_mask(offsets, stage2_span)
        stage3_mask = _char_spans_to_token_mask(offsets, stage3_span)
        answer_mask = _char_spans_to_token_mask(offsets, answer_span)

        # Enforce disjoint stage masks so token-level stage accuracies are
        # interpretable and non-overlapping.
        stage2_mask = stage2_mask & ~stage3_mask
        stage1_mask = stage1_mask & ~stage2_mask & ~stage3_mask
        answer_mask = answer_mask & stage3_mask

        if int(stage3_mask.sum().item()) == 0:
            filtered_missing_stage3 += 1
            continue

        if int(stage2_mask.sum().item()) > 0:
            stage2_non_empty += 1
        stage3_non_empty += 1
        if int(answer_mask.sum().item()) > 0:
            valid_answer_spans += 1

        labels = input_ids.clone()
        final_answer_text = answer.strip()
        numeric_values = extract_numeric_values(final_answer_text)
        if numeric_values:
            numeric_answers += 1

        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "stage1_mask": stage1_mask,
                "stage2_mask": stage2_mask,
                "stage3_mask": stage3_mask,
                "answer_mask": answer_mask,
                "final_answer_mask": answer_mask,
                "answer_text": final_answer_text,
                "answer_text_normalized": normalize_answer_text(final_answer_text),
                "source_signature": normalize_answer_text(problem, semantic_numeric=False) + " || " + normalize_answer_text(final_answer_text, semantic_numeric=False),
            }
        )

    if not examples:
        raise RuntimeError("No valid examples were built from MetaMathQA.")

    stats = {
        "raw_selected_examples": raw_selected,
        "kept_examples": len(examples),
        "filtered_examples_total": filtered_empty + filtered_missing_stage3,
        "filtered_examples_empty_problem_or_response": filtered_empty,
        "filtered_examples_missing_stage3_tokens": filtered_missing_stage3,
        "examples_with_non_empty_stage2": stage2_non_empty,
        "examples_with_non_empty_stage3": stage3_non_empty,
        "samples_with_valid_answer_spans": valid_answer_spans,
        "samples_with_numeric_answers": numeric_answers,
        "samples_excluded_or_degraded": filtered_empty + filtered_missing_stage3 + (len(examples) - valid_answer_spans),
        "samples_truncated_to_max_seq_length": truncated_examples,
        "samples_with_answer_span_truncated": answer_span_truncated,
    }
    return examples, stats


def build_external_examples_from_hf(
    *,
    dataset_name: str,
    split: str,
    subset_size: int,
    seed: int,
    max_seq_length: int,
    tokenizer: Any,
    cache_dir: str | None,
) -> tuple[list[Example], dict[str, int]]:
    """Build answer-span-evaluable examples for external datasets."""
    from datasets import load_dataset  # type: ignore

    key = dataset_name.lower()
    if key == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
        question_key, answer_key = "question", "answer"
        extract_answer = lambda value: str(value).split("####")[-1].strip()  # noqa: E731
    elif key == "math":
        ds = load_dataset("competition_math", split=split, cache_dir=cache_dir)
        question_key, answer_key = "problem", "solution"
        extract_answer = lambda value: _extract_reasoning_and_answer(str(value))[1].strip()  # noqa: E731
    elif key == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split=split, cache_dir=cache_dir)
        question_key, answer_key = "Question", "Answer"
        extract_answer = lambda value: str(value).strip()  # noqa: E731
    else:
        raise ValueError(f"Unsupported external evaluation dataset '{dataset_name}'.")

    order = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(order)
    selected = order[:subset_size] if subset_size > 0 else order

    examples: list[Example] = []
    truncated = 0
    answer_truncated = 0
    dropped = 0
    numeric_answers = 0
    for idx in selected:
        row = ds[int(idx)]
        problem = str(row.get(question_key, "")).strip()
        raw_answer = row.get(answer_key, "")
        answer = extract_answer(raw_answer)
        if not problem or not answer:
            dropped += 1
            continue
        text, span = _build_plain_text(problem, answer)
        ex, ex_trunc, ans_trunc = _example_from_text(
            tokenizer=tokenizer,
            text=text,
            answer_span=span,
            max_seq_length=max_seq_length,
            answer_text=answer,
            source_signature=normalize_answer_text(problem, semantic_numeric=False) + " || " + normalize_answer_text(answer, semantic_numeric=False),
        )
        if ex_trunc:
            truncated += 1
        if ans_trunc:
            answer_truncated += 1
        if ex is None:
            dropped += 1
            continue
        if extract_numeric_values(answer):
            numeric_answers += 1
        examples.append(ex)

    if not examples:
        raise RuntimeError(f"No valid examples were built for external dataset '{dataset_name}'.")

    return examples, {
        "raw_selected_examples": len(selected),
        "kept_examples": len(examples),
        "filtered_examples_total": dropped,
        "samples_with_valid_answer_spans": len(examples),
        "samples_with_numeric_answers": numeric_answers,
        "samples_truncated_to_max_seq_length": truncated,
        "samples_with_answer_span_truncated": answer_truncated,
        "external_dataset_name": key,
        "external_split": split,
    }


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
        out.append(
            {
                "input_ids": ids,
                "labels": ids.clone(),
                "stage1_mask": s1,
                "stage2_mask": s2,
                "stage3_mask": s3,
                "answer_mask": s3.clone(),
                "final_answer_mask": s3.clone(),
                "answer_text": "7",
                "answer_text_normalized": normalize_answer_text("7"),
                "source_signature": f"synthetic-{seed}-{i}",
            }
        )
    return out


def collate_token_sequences(batch: list[Example], *, pad_token_id: int) -> dict[str, torch.Tensor | list[str]]:
    """Pad variable-length examples and preserve stage/answer masks."""
    max_len = max(int(item["input_ids"].shape[0]) for item in batch)
    lengths = [int(item["input_ids"].shape[0]) for item in batch]

    def _pad_1d(x: torch.Tensor, pad_val: int = 0) -> torch.Tensor:
        if x.shape[0] == max_len:
            return x
        pad = torch.full((max_len - x.shape[0],), pad_val, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([_pad_1d(item["input_ids"], pad_token_id) for item in batch])
    labels = torch.stack([_pad_1d(item["labels"], -100) for item in batch])
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    collated: dict[str, torch.Tensor | list[str]] = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    for key in ("stage1_mask", "stage2_mask", "stage3_mask"):
        collated[key] = torch.stack([_pad_1d(item[key].bool(), 0).bool() for item in batch])
    collated["answer_mask"] = torch.stack(
        [
            _pad_1d(
                (item["answer_mask"] if "answer_mask" in item else item["final_answer_mask"]).bool(),
                0,
            ).bool()
            for item in batch
        ]
    )
    collated["final_answer_mask"] = collated["answer_mask"]

    collated["answer_text"] = [str(item.get("answer_text", "")) for item in batch]
    collated["answer_text_normalized"] = [str(item.get("answer_text_normalized", "")).strip() for item in batch]
    return collated


@dataclass(slots=True)
class DatasetBundle:
    train: SequenceDataset
    eval: SequenceDataset
    preprocessing_summary: dict[str, Any]


def _stable_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sample_ids_hash(examples: list[Example]) -> str:
    sample_ids = [str(ex.get("source_signature", "")) for ex in examples]
    return _stable_hash(sample_ids)


def _dataset_identity_payload(
    *,
    dataset_name: str,
    upstream_split: str,
    dataset_seed: int,
    preprocessing_settings: dict[str, Any],
    selected_sample_ids: list[str],
    train_examples: list[Example],
    eval_examples: list[Example],
) -> dict[str, Any]:
    """Build stable dataset identity fields for reproducibility and pairing.

    `dataset_fingerprint` summarizes preprocessing settings plus sampled IDs.
    Train/eval sample hashes are also broken out explicitly to allow strict
    paired-run checks on evaluation cohorts.
    """
    train_hash = _sample_ids_hash(train_examples)
    eval_hash = _sample_ids_hash(eval_examples)
    payload = {
        "dataset_name": dataset_name,
        "upstream_split": upstream_split,
        "dataset_seed": int(dataset_seed),
        "preprocessing_settings": preprocessing_settings,
        "selected_sample_ids_hash": _stable_hash(selected_sample_ids),
        "selected_sample_count": len(selected_sample_ids),
        "train_sample_ids_hash": train_hash,
        "eval_sample_ids_hash": eval_hash,
        "train_sample_count": len(train_examples),
        "eval_sample_count": len(eval_examples),
    }
    payload["dataset_fingerprint"] = _stable_hash(payload)
    return payload


def _split_counts(total_examples: int, eval_fraction: float) -> tuple[int, int]:
    eval_examples = max(1, int(total_examples * eval_fraction))
    train_examples = max(1, total_examples - eval_examples)
    return train_examples, eval_examples


def build_train_eval_datasets(name: str, settings: dict[str, Any], vocab_size: int, tokenizer: Any | None = None) -> DatasetBundle:
    """Build primary train/eval datasets and preprocessing summary metadata."""
    total_examples = int(settings.get("subset_size", 25_000))
    eval_fraction = float(settings.get("eval_fraction", 0.1))
    seed = int(settings.get("seed", 0))
    train_count, eval_count = _split_counts(total_examples, eval_fraction)

    if name == "test_synthetic_stage_dataset":
        seq_len = int(settings.get("sequence_length", 12))
        effective_vocab_size = max(8, int(vocab_size))
        train = build_test_examples(num_examples=train_count, sequence_length=seq_len, vocab_size=effective_vocab_size, seed=seed)
        evalv = build_test_examples(num_examples=eval_count, sequence_length=seq_len, vocab_size=effective_vocab_size, seed=seed + 1)
        summary = {
            "raw_selected_examples": total_examples,
            "kept_examples": total_examples,
            "filtered_examples_total": 0,
            "filtered_examples_empty_problem_or_response": 0,
            "filtered_examples_missing_stage3_tokens": 0,
            "examples_with_non_empty_stage2": total_examples,
            "examples_with_non_empty_stage3": total_examples,
            "samples_with_valid_answer_spans": total_examples,
            "samples_with_numeric_answers": total_examples,
            "samples_excluded_or_degraded": 0,
            "samples_truncated_to_max_seq_length": 0,
            "samples_with_answer_span_truncated": 0,
        }
        identity = _dataset_identity_payload(
            dataset_name=name,
            upstream_split=str(settings.get("split", "synthetic")),
            dataset_seed=seed,
            preprocessing_settings={k: settings.get(k) for k in sorted(settings)},
            selected_sample_ids=[str(ex.get("source_signature", "")) for ex in (train + evalv)],
            train_examples=train,
            eval_examples=evalv,
        )
        summary.update(identity)
        summary.update({
            "dataset_split": str(settings.get("split", "synthetic")),
            "dataset_seed": seed,
            "dataset_subset_size": total_examples,
            "dataset_eval_fraction": eval_fraction,
        })
        return DatasetBundle(train=SequenceDataset(train), eval=SequenceDataset(evalv), preprocessing_summary=summary)

    if name != "metamath_qa":
        raise ValueError(f"Unsupported dataset '{name}'.")
    if tokenizer is None:
        raise ValueError("Tokenizer is required for metamath_qa dataset.")

    examples, preprocessing_summary = build_staged_examples_from_hf(
        subset_size=total_examples,
        seed=seed,
        max_seq_length=int(settings.get("max_seq_length", 2048)),
        tokenizer=tokenizer,
        cache_dir=settings.get("cache_dir"),
        split=str(settings.get("split", "train")),
    )
    train_examples = examples[:train_count]
    train_signatures = {str(ex.get("source_signature", "")) for ex in train_examples}
    eval_candidates = examples[train_count:]
    eval_examples: list[Example] = []
    dropped_signature_overlap = 0
    for ex in eval_candidates:
        sig = str(ex.get("source_signature", ""))
        # Avoid leakage between train/eval by removing signature duplicates.
        if sig and sig in train_signatures:
            dropped_signature_overlap += 1
            continue
        eval_examples.append(ex)
        if len(eval_examples) >= eval_count:
            break

    if not eval_examples:
        raise RuntimeError("No eval examples remain after train/eval signature de-duplication.")

    preprocessing_summary = {
        **preprocessing_summary,
        "train_eval_partition_strategy": "single_split_shuffle_then_dedup_by_problem_answer_signature",
        "train_eval_split_source": str(settings.get("split", "train")),
        "train_examples_written": len(train_examples),
        "eval_examples_written": len(eval_examples),
        "eval_candidates_dropped_signature_overlap": dropped_signature_overlap,
    }
    identity = _dataset_identity_payload(
        dataset_name=name,
        upstream_split=str(settings.get("split", "train")),
        dataset_seed=seed,
        preprocessing_settings={k: settings.get(k) for k in sorted(settings)},
        selected_sample_ids=[str(ex.get("source_signature", "")) for ex in examples],
        train_examples=train_examples,
        eval_examples=eval_examples,
    )
    preprocessing_summary.update(identity)
    preprocessing_summary.update({
        "dataset_split": str(settings.get("split", "train")),
        "dataset_seed": seed,
        "dataset_subset_size": total_examples,
        "dataset_eval_fraction": eval_fraction,
    })

    return DatasetBundle(
        train=SequenceDataset(train_examples),
        eval=SequenceDataset(eval_examples),
        preprocessing_summary=preprocessing_summary,
    )


def build_external_eval_dataset(name: str, settings: dict[str, Any], tokenizer: Any | None) -> DatasetBundle:
    """Build external evaluation dataset bundle.

    External datasets are eval-only semantically; we mirror train/eval handles
    to satisfy a common DatasetBundle interface.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for external evaluation datasets.")
    split = str(settings.get("split", "test"))
    seed = int(settings.get("seed", 0))
    subset_size = int(settings.get("subset_size", 0))
    examples, summary = build_external_examples_from_hf(
        dataset_name=name,
        split=split,
        subset_size=subset_size,
        seed=seed,
        max_seq_length=int(settings.get("max_seq_length", 2048)),
        tokenizer=tokenizer,
        cache_dir=settings.get("cache_dir"),
    )
    identity = _dataset_identity_payload(
        dataset_name=name,
        upstream_split=split,
        dataset_seed=seed,
        preprocessing_settings={k: settings.get(k) for k in sorted(settings)},
        selected_sample_ids=[str(ex.get("source_signature", "")) for ex in examples],
        train_examples=examples,
        eval_examples=examples,
    )
    summary.update(identity)
    summary.update({
        "dataset_split": split,
        "dataset_seed": seed,
        "dataset_subset_size": subset_size,
        "dataset_eval_fraction": 0.0,
        "dataset_type": "external",
    })
    return DatasetBundle(train=SequenceDataset(examples), eval=SequenceDataset(examples), preprocessing_summary=summary)
