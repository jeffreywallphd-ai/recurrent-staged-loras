# Documentation Overview

This repository is organized as a **controlled empirical comparison study** of latent adaptation strategies under a fixed base-model setup.

## Purpose

The project compares five baselines on the same base instruct model and dataset/task setup:

1. Base model (no adaptation)
2. Base + standard LoRA
3. Base + latent refiner only
4. Base + latent refiner with shared recurrence
5. Base + latent refiner with stage-specialized recurrence

The immediate goal is to provide a clean implementation scaffold for these comparisons, not to introduce or claim a fundamentally novel architecture.

## Current phase

This version intentionally focuses on:

- module boundaries and interfaces,
- experiment configuration templates,
- consistent baseline definitions,
- documentation for controlled comparisons.

It intentionally does **not** include full training logic, finalized refiner internals, or full-scale benchmark claims.

## Reading order

- `docs/architecture.md`: planned system layout and dataflow.
- `docs/baselines.md`: baseline definitions and hypotheses.
- `docs/experiments.md`: initial experiment matrix and success/failure criteria.
