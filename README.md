# recurrent-staged-loras

A configurable, end-to-end experiment system for controlled adaptation studies across **dense** and **MoE** backbones.

## Default study backbones

- Dense default: `Qwen/Qwen3-8B`
- MoE default: `allenai/OLMoE-1B-7B-0125-Instruct`

Both are first-class defaults in `experiments/configs/` and share the same baseline family:
1. `base`
2. `standard_lora`
3. `latent_refiner_only`
4. `shared_recurrence`
5. `stage_specialized_recurrence`

## Real dataset default

Default dataset is `metamath_qa` (`meta-math/MetaMathQA`) with deterministic subsetting (`subset_size`, `seed`) and local caching (`cache_dir`).

Each sample is converted into explicit 3-stage supervision:
- Stage 1: problem understanding region
- Stage 2: intermediate reasoning region
- Stage 3: final answer region

Stage masks are materialized in dataset examples and passed through collation (`stage1_mask`, `stage2_mask`, `stage3_mask`).

## Real model configuration schema

`model` supports:
- `name`
- `tokenizer_name`
- `trust_remote_code`
- `dtype`
- `device_map`
- `max_seq_length`
- `load_in_4bit`
- `bnb_4bit_compute_dtype`
- `attn_implementation`
- `gradient_checkpointing`
- `architecture_type` (`dense` or `moe`)

## Training + evaluation protocol

- Base model is frozen in HF path.
- Standard LoRA uses real PEFT target-module injection.
- Recurrent modes use explicit stage-aware loss (step-to-stage alignment).
- Evaluation records loss/perplexity, stage token accuracies, final answer metrics, numeric-answer proxy accuracy, and compute fairness metrics.

## Multi-seed batch runs

`run_all_experiments.py` supports `--seeds` (default `11 22 33`) and writes:
- per-run `metrics.json`
- `outputs/summary.json` with `runs` + `aggregates` (mean/std)
- `outputs/summary.csv`

## Quick start

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name dense_lora_seed11
python scripts/run_all_experiments.py --configs experiments/configs/base.json experiments/configs/moe_base.json --seeds 11 22 33
```
