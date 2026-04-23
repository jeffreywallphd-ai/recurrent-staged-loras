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

By default, dataset artifacts are cached under `./.cache/hf_datasets` unless overridden in config.

## Quick Start (setup verification in minutes)

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
