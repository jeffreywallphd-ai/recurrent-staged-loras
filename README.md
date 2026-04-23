# recurrent-staged-loras

Core scaffolding for a controlled empirical comparison of latent adaptation baselines on top of a frozen base language model.

Current phase: establish modular interfaces, experiment configs, and baseline documentation for a controlled empirical comparison study.

## Rationale

This repository isolates adaptation choices while holding the base model, dataset protocol, and evaluation framing fixed. The goal is to make baseline-to-baseline comparisons explicit and reproducible before full training internals are implemented.

## Baselines in scope

1. Base
2. Base + standard LoRA
3. Base + latent refiner only
4. Base + latent refiner + shared recurrence
5. Base + latent refiner + stage-specialized recurrence

Baseline templates live in `experiments/configs/` and are validated by lightweight smoke tests.

## Architecture path

Planned composition for latent-refiner variants:

**frozen base LM -> latent refiner -> LM head**

The latent refiner path can be configured with no adapters, shared recurrence adapters, or stage-specialized recurrence adapters.

## Status (scaffold-only)

This version is intentionally scaffold-focused:

- baseline definitions and naming discipline,
- config wiring and baseline selection,
- run metadata plumbing,
- lightweight smoke coverage.

It does **not** yet implement the full latent refiner internals, full adapter backends, or full training loop behavior.

## Read next

- `docs/README.md` for project framing and reading order.
- `docs/architecture.md` for component layout and mode definitions.
- `docs/baselines.md` for detailed baseline hypotheses.
- `docs/experiments.md` for the experiment matrix and scaffold criteria.

## Repository layout

- `models/`: model wrapper interfaces and latent refiner composition.
- `training/`: config loading, baseline selection, run metadata, training entrypoint scaffold.
- `data/`: dataset plumbing placeholders.
- `experiments/configs/`: baseline experiment templates.
- `docs/`: architecture and empirical framing docs.
- `scripts/`: convenience launch scripts.
- `tests/`: minimal smoke tests for scaffolding.

## Quick start (scaffold)

```bash
python -m training.train --config experiments/configs/base.json --run-name smoke_base
```

This currently writes run metadata and validates config wiring. Full training implementation is intentionally deferred.
