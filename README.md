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
output/<baseline>/<config_stem>/
  config.json
  metadata.json
  metrics.json
  checkpoint.pt

output/summary.json
```

### Compare metrics from runs

```bash
python scripts/compare_metrics.py output/base/base/metrics.json output/standard_lora/standard_lora/metrics.json
```

`metrics.json` includes:
- `baseline_name`
- `train_loss`
- `eval_loss` (if eval enabled)
- `num_steps`
- `num_epochs`

Use `output/summary.json` as the baseline-to-metrics map for quick comparisons.

## Reading order

1. `docs/README.md`
2. `docs/architecture.md`
3. `docs/baselines.md`
4. `docs/experiments.md`
