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

`metrics.json` includes:
- `baseline_name`
- `train_loss`
- `eval_loss` (if enabled)
- `num_steps`
- `num_epochs`

## Running one experiment

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name run01
```

## Running many experiments

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs
```

This writes per-baseline runs under `output/<baseline>/<config_stem>/` and aggregates to `output/summary.json`.

## Comparison process

For quick terminal comparisons:

```bash
python scripts/compare_metrics.py output/base/base/metrics.json output/standard_lora/standard_lora/metrics.json
```

This prints a compact table over baseline name and key metric fields.
