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
- dataset abstraction with deterministic local modes (`synthetic_integer_sequences`, `text_style_patterns`),
- deterministic run artifacts (`config.json`, `metadata.json`, `metrics.json`, `checkpoint.pt`).

Latent cache behavior is explicitly disabled for now and reserved as future work.

## Running experiments

Run a config with:

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name smoke_standard_lora
```

Output structure is standardized as:

```text
outputs/<baseline>/<run_name>/
  config.json
  metadata.json
  metrics.json
  checkpoint.pt
```

`metrics.json` includes `train_loss`, `eval_loss`, and step/epoch counts for quick run validation.

## Reading order

1. `docs/README.md`
2. `docs/architecture.md`
3. `docs/baselines.md`
4. `docs/experiments.md`
