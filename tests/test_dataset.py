import torch

from data.dataset import _build_staged_text, build_external_eval_dataset, build_train_eval_datasets, collate_token_sequences
from training.answer_eval import (
    NUMERIC_MULTI_VALUE_RULE,
    extract_numeric_values,
    normalize_answer_text,
    numeric_match,
    symbolic_equivalence_match,
)


def test_stage_masks_present_and_valid_in_test_dataset() -> None:
    bundle = build_train_eval_datasets(
        name="test_synthetic_stage_dataset",
        settings={"subset_size": 12, "sequence_length": 12, "eval_fraction": 0.25, "seed": 5},
        vocab_size=128,
    )
    ex = bundle.train[0]
    assert "stage1_mask" in ex and "stage2_mask" in ex and "stage3_mask" in ex and "answer_mask" in ex
    assert int((ex["stage1_mask"] | ex["stage2_mask"] | ex["stage3_mask"]).sum().item()) == ex["input_ids"].shape[0]

    batch = collate_token_sequences([bundle.train[0], bundle.train[1]], pad_token_id=9)
    assert batch["stage1_mask"].shape == batch["input_ids"].shape
    assert batch["stage2_mask"].shape == batch["input_ids"].shape
    assert batch["stage3_mask"].shape == batch["input_ids"].shape
    assert batch["answer_mask"].shape == batch["input_ids"].shape


def test_tokenizer_aware_padding_uses_pad_token_id_and_keeps_alignment() -> None:
    batch = [
        {
            "input_ids": torch.tensor([5, 6, 7], dtype=torch.long),
            "labels": torch.tensor([5, 6, 7], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1, 0], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 0, 1], dtype=torch.bool),
            "answer_mask": torch.tensor([0, 0, 1], dtype=torch.bool),
            "answer_text": "7",
            "answer_text_normalized": "7",
        },
        {
            "input_ids": torch.tensor([8, 9], dtype=torch.long),
            "labels": torch.tensor([8, 9], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 0], dtype=torch.bool),
            "answer_mask": torch.tensor([0, 0], dtype=torch.bool),
            "answer_text": "9",
            "answer_text_normalized": "9",
        },
    ]
    collated = collate_token_sequences(batch, pad_token_id=99)
    assert collated["input_ids"][1, 2].item() == 99
    assert collated["attention_mask"][1, 2].item() == 0
    assert collated["labels"][1, 2].item() == -100
    assert bool(collated["stage1_mask"][1, 2].item()) is False
    assert bool(collated["stage2_mask"][1, 2].item()) is False
    assert bool(collated["stage3_mask"][1, 2].item()) is False
    assert bool(collated["answer_mask"][1, 2].item()) is False


def test_attention_mask_uses_sequence_lengths_not_pad_token_value() -> None:
    batch = [
        {
            "input_ids": torch.tensor([5, 99, 7], dtype=torch.long),
            "labels": torch.tensor([5, 99, 7], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1, 0], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 0, 1], dtype=torch.bool),
            "answer_mask": torch.tensor([0, 0, 1], dtype=torch.bool),
            "answer_text": "7",
            "answer_text_normalized": "7",
        },
        {
            "input_ids": torch.tensor([8, 9], dtype=torch.long),
            "labels": torch.tensor([8, 9], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 1], dtype=torch.bool),
            "answer_mask": torch.tensor([0, 1], dtype=torch.bool),
            "answer_text": "9",
            "answer_text_normalized": "9",
        },
    ]
    collated = collate_token_sequences(batch, pad_token_id=99)
    assert collated["attention_mask"][0, 1].item() == 1
    assert collated["attention_mask"][1, 2].item() == 0


def test_answer_span_definition_excludes_final_answer_header() -> None:
    full, _s1, _s2, stage3, answer = _build_staged_text("Q?", "Because", "42")
    assert full[stage3[0] : answer[0]] == "Final Answer:\n"
    assert full[answer[0] : answer[1]] == "42"


def test_normalization_and_numeric_extraction_are_reviewer_grade() -> None:
    assert normalize_answer_text("  $\\boxed{2.0}$  ") == "2"
    assert normalize_answer_text("1/2") == "0.5"
    assert extract_numeric_values("vals: -3, 2.5e1, 3/4") == [-3.0, 25.0, 0.75]


def test_numeric_matching_tolerance_and_strict_multi_value_default() -> None:
    assert NUMERIC_MULTI_VALUE_RULE == "strict_set"
    close = numeric_match("0.5000000001", "1/2")
    assert close.is_match is True

    same_set_diff_order = numeric_match("2, 1", "1, 2")
    assert same_set_diff_order.is_match is True
    assert same_set_diff_order.multi_value_status == "exact_set_match"

    partial = numeric_match("1", "1, 2")
    assert partial.is_match is False
    assert partial.multi_value_status == "partial_overlap"


def test_numeric_matching_rule_variants_are_explicit() -> None:
    assert numeric_match("1", "1, 2", multi_value_rule="any").is_match is True
    assert numeric_match("1", "1, 2", multi_value_rule="subset").is_match is True


def test_symbolic_equivalence_success_and_parse_failure() -> None:
    eq = symbolic_equivalence_match("x + x", "2*x")
    assert eq.attempted is True
    if eq.parse_success:
        assert eq.is_match is True
    else:
        assert eq.is_match is False

    fail = symbolic_equivalence_match("x + ???", "2*x")
    assert fail.attempted is True
    assert fail.parse_success is False
    assert fail.is_match is False


def test_external_dataset_loader_builds_answer_mask(monkeypatch) -> None:
    class _FakeTokenizer:
        def __call__(self, text, truncation, max_length, return_offsets_mapping, add_special_tokens):
            del truncation, max_length, return_offsets_mapping, add_special_tokens
            ids = [min(ord(c), 120) for c in text]
            return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    fake_rows = [{"question": "2+2=?", "answer": "#### 4"}, {"question": "3+3=?", "answer": "#### 6"}]

    def _fake_loader(*_args, **_kwargs):
        return fake_rows

    monkeypatch.setattr("datasets.load_dataset", _fake_loader)
    bundle = build_external_eval_dataset("gsm8k", {"split": "test", "subset_size": 2, "seed": 0, "max_seq_length": 64}, _FakeTokenizer())
    assert len(bundle.eval) == 2
    ex = bundle.eval[0]
    assert int(ex["answer_mask"].sum().item()) > 0
    assert ex["answer_text"] in {"4", "6"}


def test_external_dataset_subset_size_split_and_seed_are_respected(monkeypatch) -> None:
    class _FakeTokenizer:
        def __call__(self, text, truncation, max_length, return_offsets_mapping, add_special_tokens):
            del truncation, max_length, return_offsets_mapping, add_special_tokens
            return {"input_ids": [1] * len(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    observed: dict[str, object] = {}
    fake_rows = [{"question": f"{i}+{i}=?", "answer": f"#### {2*i}"} for i in range(10)]

    def _fake_loader(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return fake_rows

    monkeypatch.setattr("datasets.load_dataset", _fake_loader)
    bundle = build_external_eval_dataset("gsm8k", {"split": "test", "subset_size": 3, "seed": 9, "max_seq_length": 64}, _FakeTokenizer())
    assert observed["args"] == ("gsm8k", "main")
    assert observed["kwargs"] == {"split": "test", "cache_dir": None}
    assert len(bundle.eval) == 3
    assert bundle.preprocessing_summary["dataset_split"] == "test"
    assert bundle.preprocessing_summary["dataset_seed"] == 9
    assert bundle.preprocessing_summary["dataset_subset_size"] == 3
