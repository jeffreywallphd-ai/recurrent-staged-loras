# recurrent-staged-loras

Core scaffolding for a controlled empirical comparison of latent adaptation baselines on top of a frozen base language model.

## Why this repo exists

This repository isolates adaptation choices while holding base model family, dataset protocol, and evaluation framing fixed. The objective is reproducible baseline-to-baseline comparison, not novelty claims.

## Baselines in scope

1. Base
2. Base + standard LoRA
3. Base + latent refiner only
4. Base + latent refiner + shared recurrence adapters
5. Base + latent refiner + stage-specialized recurrence adapters

Baseline templates live in `experiments/configs/`.

## Architecture path

**frozen base LM -> latent refiner -> LM head**

The latent refiner path is configurable for:
- recurrence with no adapters (`latent_refiner_only`),
- recurrence with one shared adapter across steps (`shared_recurrence`),
- recurrence with per-step adapters (`stage_specialized_recurrence`).

## Current status

This version provides a minimal reproducible experiment harness:
- baseline definitions and config semantics,
- model composition with torch modules and trainability checks,
- reusable training engine + loop primitives,
- dataset abstraction with deterministic local modes (`synthetic_integer_sequences`, `text_style_patterns`, `structured_sequence`),
- deterministic run artifacts (`config.json`, `metadata.json`, `metrics.json`, `checkpoint.pt`).

Latent caching is deferred to future work and is intentionally disabled in the active training path.

## Running experiments

### Run a single config

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name smoke_standard_lora
```

Output structure for single runs:

```text
outputs/<baseline>/<run_name>/
  config.json
  metadata.json
  metrics.json
  checkpoint.pt
```

### Run multiple configs

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs
# or
python scripts/run_all_experiments.py --configs experiments/configs/base.json experiments/configs/standard_lora.json
```

Output structure for multi-run mode:

```text
outputs/<baseline>/<config_stem>/
  config.json
  metadata.json
  metrics.json
  checkpoint.pt

outputs/summary.json
outputs/summary.csv
```

### Compare metrics from runs

```bash
python scripts/compare_metrics.py outputs/base/base/metrics.json outputs/standard_lora/standard_lora/metrics.json
```

`metrics.json` includes reproducibility context, benchmark outcome metrics, and compute/efficiency metrics.
Core fields:
- Run/context: `run_name`, `config_name`, `baseline_name`, `dataset_name`, `dataset_mode`, `dataset_train_examples`, `dataset_eval_examples`, `batch_size`, `learning_rate`, `weight_decay`, `seed`, `deterministic`, `backend`
- Outcome metrics: `final_train_loss`, `final_eval_loss`, `best_eval_loss`, `eval_perplexity`, `train_perplexity`, `eval_next_token_accuracy`, `eval_top_5_accuracy`, `global_steps_completed`, `epochs_completed`
- Task-specific metrics (structured sequence only): `eval_target_token_accuracy`, `eval_target_sequence_exact_match`
- Compute/efficiency: `wall_time_seconds_total`, `wall_time_seconds_train`, `wall_time_seconds_eval`, `tokens_seen_train`, `tokens_seen_eval`, `tokens_per_second_train`, `tokens_per_second_eval`, `seconds_per_step`, `steps_per_second`
- Parameterization: `trainable_params`, `total_params`, `trainable_param_fraction`
- Compatibility aliases: `train_loss`, `eval_loss`, `num_steps`, `num_epochs`

Timing semantics are strict:
- `wall_time_seconds_train` excludes all evaluation passes.
- `wall_time_seconds_eval` includes interval and final evaluation.
- `tokens_seen_eval` includes interval and final evaluation tokens.

`outputs/summary.json` stores `{"runs": [...]}` where each item is one run record, including identifiers, paths, and copied comparison metrics.
`outputs/summary.csv` provides a deterministic tabular version for downstream empirical analysis.

## Reading order

1. `docs/README.md`
2. `docs/architecture.md`
3. `docs/baselines.md`
4. `docs/experiments.md`
