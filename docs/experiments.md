# Experiment Workflow

## Objective

Run controlled baseline comparisons with standardized metrics and reproducible outputs.

## Dataset modes

Supported local dataset modes:
- `synthetic_integer_sequences`
- `text_style_patterns`
- `structured_sequence`

All modes emit labeled examples (`input_ids`, `labels`) using next-token prediction labels.
`structured_sequence` additionally emits `target_mask`, which marks the target span used for task-specific eval metrics.

## Reproducibility protocol

Training config supports:
- `seed` for Python random + PyTorch + dataset generation
- `deterministic` for deterministic PyTorch algorithm preference

Notes:
- Deterministic settings may reduce performance and can still vary across hardware/backends.

## Training + evaluation

The training loop computes:
- train loss during training
- evaluation loss at end of training
- optional interval evaluation during training (`eval_interval_steps` + `eval_enabled`)

Timing semantics:
- `wall_time_seconds_train` measures pure training step time (interval eval excluded)
- `wall_time_seconds_eval` accumulates all eval passes (interval + final)
- `wall_time_seconds_total` is full run wall clock

## Required run artifacts

Each run produces:
- `config.json`
- `metadata.json`
- `metrics.json`
- `checkpoint.pt`

`metrics.json` schema:

Run/context fields:
- `run_name`
- `config_name`
- `baseline_name`
- `dataset_name`
- `dataset_mode`
- `dataset_train_examples`
- `dataset_eval_examples`
- `batch_size`
- `learning_rate`
- `weight_decay`
- `seed`
- `deterministic`
- `backend`

Outcome metrics:
- `final_train_loss`
- `final_eval_loss`
- `best_eval_loss`
- `train_perplexity`
- `eval_perplexity`
- `eval_next_token_accuracy`
- `eval_top_5_accuracy`
- `global_steps_completed`
- `epochs_completed`

Task-specific outcome metrics (only for `structured_sequence`):
- `eval_target_token_accuracy`
- `eval_target_sequence_exact_match`

Compute/efficiency fields:
- `wall_time_seconds_total`
- `wall_time_seconds_train`
- `wall_time_seconds_eval`
- `tokens_seen_train`
- `tokens_seen_eval`
- `tokens_per_second_train`
- `tokens_per_second_eval`
- `seconds_per_step`
- `steps_per_second`

Parameterization fields:
- `trainable_params`
- `total_params`
- `trainable_param_fraction`

Compatibility aliases:
- `train_loss` (same value as `final_train_loss`)
- `eval_loss` (same value as `final_eval_loss`)
- `num_steps` (same value as `global_steps_completed`)
- `num_epochs` (same value as `epochs_completed`)

Perplexity notes:
- Perplexity is derived from next-token cross-entropy (`exp(loss)`), with numeric clamping for stability.

## Running one experiment

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name run01
```

## Running many experiments

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs
```

This writes per-baseline runs under `outputs/<baseline>/<config_stem>/` and aggregates to `outputs/summary.json`.
Batch runs also produce `outputs/summary.csv`.

Batch aggregation format:
- `outputs/summary.json` is `{"runs": [...]}`.
- Each run record includes baseline, run/config identity, paths, and key outcome/compute metrics.
- Runs are append-like within a single invocation and are not collapsed by baseline key, so multiple runs for the same baseline remain visible.

## Comparison process

For quick terminal comparisons:

```bash
python scripts/compare_metrics.py outputs/base/base/metrics.json outputs/standard_lora/standard_lora/metrics.json
```

This prints a compact comparison table over quality and efficiency fields (losses, perplexity, accuracies, parameters, timing, and throughput).
