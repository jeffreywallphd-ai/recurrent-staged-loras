# recurrent-staged-loras

Controlled dense-vs-MoE adaptation study pipeline for staged supervision on MetaMathQA.

## Defaults

- Dense family: `Qwen/Qwen3-8B`
- MoE family: `allenai/OLMoE-1B-7B-0125-Instruct`
- Baselines: `base`, `standard_lora`, `latent_refiner_only`, `shared_recurrence`, `stage_specialized_recurrence`
- Dataset: `meta-math/MetaMathQA` with explicit 3-stage masks.

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/jeffreywallphd-ai/recurrent-staged-loras.git
cd recurrent-staged-loras
```

### 2) Python version

Use **Python 3.10+** (Python 3.10 or 3.11 recommended).

### 3) Create and activate a virtual environment (venv preferred)

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Optional Conda alternative:

```bash
conda create -n recurrent-staged-loras python=3.10 -y
conda activate recurrent-staged-loras
python -m pip install --upgrade pip
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

For contributors running tests:

```bash
pip install -r requirements-dev.txt
```

### 5) Optional Hugging Face authentication

For gated/rate-limited model or dataset access, authenticate once:

```bash
huggingface-cli login
```
or set an environment variable for non-interactive shells:
```bash
export HF_TOKEN=hf_xxx
```

By default, dataset artifacts are cached under `./.cache/hf_datasets` unless overridden in config.
Never place tokens in repository configs (`*.json`) or commit token files.

## Optional run publishing to Hugging Face Hub

Publishing is opt-in and disabled by default. Add a `publish` block to an experiment config:

```json
"publish": {
  "enabled": true,
  "hub_model_repo": "org-or-user/recurrent-staged-loras-model",
  "hub_dataset_repo": "org-or-user/recurrent-staged-loras-dataset",
  "private": true,
  "commit_message": "Publish run artifacts",
  "include_checkpoint": false,
  "include_metrics": true,
  "include_dataset_partitions": true
}
```

Uploaded model-side artifacts include safetensors model weights (`model.safetensors` or sharded `model-*.safetensors` + index), `config.json`, `metadata.json`, `metrics.json`, `answer_eval_diagnostics.json`, `dataset_preprocessing_summary.json`, and a generated model card. Local `checkpoint.pt` files are removed after safetensors validation during publish.
Uploaded dataset-side artifacts include exact `train`/`eval` partitions, sample IDs/hashes/fingerprints, preprocessing metadata, and a dataset card.

Publish an existing completed run directory:

```bash
python -m scripts.publish_run_to_hf \
  --run-dir outputs/local_synthetic_debug/local_synthetic_debug_seed7 \
  --model-repo org-or-user/recurrent-staged-loras-model \
  --dataset-repo org-or-user/recurrent-staged-loras-dataset \
  --private \
  --commit-message \"Publish debug run\"
```

Security reminders:
- Never commit tokens.
- Never put tokens in JSON experiment configs.
- Verify rights for any dataset content before publishing.

## Quick Start (setup verification in minutes)

Simple train command:

```bash
python -m training.train --config experiments/configs/stage_specialized_recurrence_debug.json --run-name smoke_standard_lora
```

Run a local CPU-friendly debug preset via the experiment orchestrator:

```bash
python -m scripts.run_all_experiments \
  --configs experiments/configs/local_synthetic_debug.json \
  --config-family debug \
  --preset-scope all \
  --run-scope confirmatory \
  --seeds 7
```

What to expect:

- A run directory is written under `outputs/local_synthetic_debug/` (for example `outputs/local_synthetic_debug/local_synthetic_debug_seed7/`).
- Per-run artifacts include:
  - `metrics.json`
  - `dataset_preprocessing_summary.json`
  - `answer_eval_diagnostics.json`
- Aggregated artifacts are written to `outputs/`:
  - `summary.json`
  - `summary.csv`
  - `aggregates.json`
  - `report_table.csv`

**Installation success check:** if `outputs/summary.json` and `outputs/local_synthetic_debug/.../metrics.json` exist after the command, your environment is set up correctly.

## GPU / Hardware Guidance

- **GPU is optional** for the local synthetic debug preset above.
- **GPU strongly recommended** for full MetaMathQA and large model presets.

PyTorch install notes:

- The default `pip install -r requirements.txt` installs PyTorch from your configured index.
- If you need a specific build (CPU-only or CUDA), install PyTorch first from the official selector and then install the rest:

```bash
# Example pattern only; choose your exact command from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Runtime expectations:

- Local synthetic debug preset: typically a few minutes on CPU.
- Full study configs: substantially longer and usually require GPU memory/throughput suitable for 8B-class models.

### Model loading modes (lab GPU vs constrained laptop)

Large backbones can run in two practical modes:

- `model.model_loading.mode = "full_gpu"`: asks Transformers to materialize the model without auto offload/sharding.
- `model.model_loading.mode = "auto"`: keeps `device_map` behavior for capability-aware loading; this may offload shards on constrained machines.

For staged/recurrent training, this repo directly invokes the base LM head (`forward_lm_head`) after refinement. Direct submodule calls are unsafe when that submodule remains on `meta` due to offload dispatch assumptions. Startup validation now fails early with actionable parameter/module names if meta tensors are detected in required paths.

## Troubleshooting

- **Missing dependencies / import errors**
  - Confirm the venv is active and rerun `pip install -r requirements.txt`.
  - For tests/tools, also install `requirements-dev.txt`.

- **Hugging Face download/auth errors**
  - Run `huggingface-cli login` and verify internet access.
  - Retry if you hit transient network/rate-limit failures.

- **CUDA/GPU mismatch**
  - Reinstall a PyTorch build matching your CUDA/runtime stack.
  - If needed, use CPU-only PyTorch for debug validation.

- **Runs are unexpectedly slow**
  - Use a `*_debug.json` preset first.
  - Keep `subset_size` and `max_steps` small for smoke checks.
- **Meta tensor / offload errors (for example `Cannot copy out of meta tensor; no data!`)**
  - Keep `model.model_loading.require_no_meta_for_training=true` to fail before the first batch with clear diagnostics.
  - Prefer a smaller model or `model.model_loading.mode="full_gpu"` when hardware can fully materialize required modules.
  - Avoid direct calls into offloaded HF submodules unless those submodules are guaranteed materialized.

## Study outputs

Each run writes:
- `metrics.json`
- `dataset_preprocessing_summary.json`
- `answer_eval_diagnostics.json`

Multi-seed orchestration writes:
- `outputs/summary.json` (runs + aggregates)
- `outputs/summary.csv` (run and aggregate rows)
- `outputs/aggregates.json` (aggregate-only artifact)
- `outputs/report_table.csv` (canonical paper/report table source with explicit run vs aggregate row typing)
- `outputs/statistical_analysis_confirmatory.json` / `.csv` (confirmatory primary-outcome contrasts with Holm-adjusted p-values)
- `outputs/statistical_analysis_secondary.json` and `outputs/statistical_analysis_efficiency.json` (separate descriptive families)
- `outputs/statistical_analysis_report.md` (reviewer-facing inferential summary)

## Presets

- Study presets: `experiments/configs/*.json`
- Pilot/smoke presets: `experiments/configs/*_pilot.json`
- Default orchestration scope is study-only (`--preset-scope study`).

## Docs

- `docs/README.md`
- `docs/architecture.md`
- `docs/baselines.md`
- `docs/experiments.md` (staging construction + strict/normalized/symbolic/numeric answer-eval protocol)
- `docs/study_matrix.md`
- `analysis/README.md`
