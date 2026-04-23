# Experiment Workflow

## Objective

Run controlled baseline comparisons with standardized metrics and reproducible outputs.

## Dataset modes

Supported local dataset modes:
- `synthetic_integer_sequences`
- `text_style_patterns`
- `structured_sequence`

All modes emit labeled examples (`input_ids`, `labels`) using next-token prediction labels.

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

Outcome fields:
- `final_train_loss`
- `final_eval_loss`
- `best_eval_loss`
- `global_steps_completed`
- `epochs_completed`

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

## Running one experiment

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name run01
```

## Running many experiments

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs
```

This writes per-baseline runs under `output/<baseline>/<config_stem>/` and aggregates to `output/summary.json`.
Batch runs also produce `output/summary.csv`.

Batch aggregation format:
- `output/summary.json` is `{"runs": [...]}`.
- Each run record includes baseline, run/config identity, paths, and key outcome/compute metrics.
- Runs are append-like within a single invocation and are not collapsed by baseline key, so multiple runs for the same baseline remain visible.

## Comparison process

For quick terminal comparisons:

```bash
python scripts/compare_metrics.py output/base/base/metrics.json output/standard_lora/standard_lora/metrics.json
```

This prints a compact comparison table over both quality and efficiency fields (losses, steps, parameters, timing, and throughput).
