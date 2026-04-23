"""Dataset abstraction tests."""

from __future__ import annotations

from data.dataset import build_train_eval_datasets, collate_token_sequences


def test_synthetic_dataset_construction_and_shapes() -> None:
    bundle = build_train_eval_datasets(
        name="synthetic_integer_sequences",
        settings={"num_examples": 20, "sequence_length": 8, "eval_fraction": 0.25, "seed": 5},
        vocab_size=128,
    )
    assert len(bundle.train) == 15
    assert len(bundle.eval) == 5

    batch = collate_token_sequences([bundle.train[0], bundle.train[1]])
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)


def test_text_style_dataset_construction_and_shapes() -> None:
    bundle = build_train_eval_datasets(
        name="text_style_patterns",
        settings={"num_examples": 12, "sequence_length": 10, "eval_fraction": 0.25, "seed": 7},
        vocab_size=128,
    )
    assert len(bundle.train) == 9
    assert len(bundle.eval) == 3

    batch = collate_token_sequences([bundle.eval[0], bundle.eval[1]])
    assert batch["input_ids"].shape == (2, 10)
    assert batch["labels"].shape == (2, 10)
    assert batch["attention_mask"].shape == (2, 10)


def test_structured_sequence_mode_construction() -> None:
    bundle = build_train_eval_datasets(
        name="structured_sequence",
        settings={"num_examples": 12, "prefix_length": 4, "target_length": 6, "eval_fraction": 0.25, "seed": 2},
        vocab_size=128,
    )
    assert len(bundle.train) == 9
    assert len(bundle.eval) == 3

    example = bundle.train[0]
    assert example["input_ids"].shape[0] == 10
    assert example["labels"].shape[0] == 10
