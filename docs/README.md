# Documentation Overview

This repository is organized as a controlled empirical comparison study of latent adaptation strategies under a fixed base-model setup.

## Purpose

The project compares five baselines on shared data/evaluation conditions:

1. Base model (no adaptation)
2. Base + standard LoRA
3. Base + latent refiner only
4. Base + latent refiner with shared recurrence adapters
5. Base + latent refiner with stage-specialized recurrence adapters

## Current phase

This version focuses on a minimal trainable baseline implementation:

- clean baseline semantics,
- model forward-path composition,
- configuration templates and typed parsing,
- lightweight forward/loss/backward smoke tests and reproducibility metadata.

It now includes a minimal local training/data/checkpoint pipeline for controlled baseline runs, standardized run artifacts (`config.json`, `metadata.json`, `metrics.json`, `checkpoint.pt`), and a small dataset abstraction with synthetic and text-style local modes; it still defers production training infrastructure and production LoRA backend integration.

## Reading order

- `docs/architecture.md`: system layout, dataflow, and mode semantics.
- `docs/baselines.md`: baseline definitions and hypotheses.
- `docs/experiments.md`: comparison matrix and scaffold-stage criteria.
