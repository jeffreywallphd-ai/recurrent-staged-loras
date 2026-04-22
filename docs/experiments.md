# Experiment Plan (Initial)

## Objective

Run a controlled comparison of latent adaptation strategies under the same frozen base model setup.

## Initial experiment matrix

| Baseline | Config file | Recurrence | Adapter scheme | Base frozen |
|---|---|---|---|---|
| Base | `experiments/configs/base.json` | No | None | Yes |
| Standard LoRA | `experiments/configs/standard_lora.json` | No | standard LoRA | Yes |
| Shared recurrence | `experiments/configs/shared_recurrence.json` | Yes (`num_recurrent_steps > 1`) | shared recurrence adapter | Yes |
| Stage-specialized recurrence | `experiments/configs/stage_specialized_recurrence.json` | Yes (`num_recurrent_steps > 1`) | per-step adapters | Yes |

## Minimum controlled variables

Across the matrix, keep fixed where feasible:

- base model checkpoint,
- tokenizer and prompt formatting,
- dataset versions and splits,
- training budget definition (tokens/steps/epochs),
- evaluation metrics and scripts,
- random seed protocol and number of repeats.

## Minimum success criteria for scaffold stage

1. Each baseline config loads successfully.
2. Baseline selector routes each config to the intended variant.
3. Training entrypoint writes run metadata for reproducibility.
4. Model interfaces are stable enough for the next step implementation.

## Failure criteria (scaffold stage)

- Baseline names/config semantics are ambiguous or inconsistent.
- Terminology drifts between "shared recurrence" and "stage-specialized recurrence".
- The architecture path is unclear about placement of the latent refiner.
- Integrating the real refiner/trainer would require major reorganization.

## Out of scope for this initial version

- Full optimizer/scheduler implementation.
- Full recurrent latent refiner internals.
- Full LoRA integration with concrete backend libraries.
- Performance claims beyond smoke-level wiring checks.
