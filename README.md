# recurrent-staged-loras

Core scaffolding for a research project that compares latent-space adaptation baselines on top of a frozen base language model.

Current phase: establish modular interfaces, experiment configs, and baseline documentation for a controlled empirical comparison study.

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
