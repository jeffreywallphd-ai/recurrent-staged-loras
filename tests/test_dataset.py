from data.dataset import build_train_eval_datasets, collate_token_sequences


def test_stage_masks_present_and_valid_in_test_dataset() -> None:
    bundle = build_train_eval_datasets(
        name="test_synthetic_stage_dataset",
        settings={"subset_size": 12, "sequence_length": 12, "eval_fraction": 0.25, "seed": 5},
        vocab_size=128,
    )
    ex = bundle.train[0]
    assert "stage1_mask" in ex and "stage2_mask" in ex and "stage3_mask" in ex
    assert int((ex["stage1_mask"] | ex["stage2_mask"] | ex["stage3_mask"]).sum().item()) == ex["input_ids"].shape[0]

    batch = collate_token_sequences([bundle.train[0], bundle.train[1]])
    assert batch["stage1_mask"].shape == batch["input_ids"].shape
    assert batch["stage2_mask"].shape == batch["input_ids"].shape
    assert batch["stage3_mask"].shape == batch["input_ids"].shape
