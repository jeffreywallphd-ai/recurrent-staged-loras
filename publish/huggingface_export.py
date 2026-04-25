"""Utilities for optional run artifact export/publish to Hugging Face Hub."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import hashlib
import json
import os
import shutil

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from training.config_loader import PublishConfig, RuntimeConfig
from training.model_validation import validate_model_checkpoint
from publish.model_serialization import serialize_checkpoint_to_hf_directory

MODEL_ARTIFACT_CANDIDATES = [
    "config.json",
    "metadata.json",
    "metrics.json",
    "answer_eval_diagnostics.json",
    "dataset_preprocessing_summary.json",
]


def _require_hf_auth() -> None:
    token = os.getenv("HF_TOKEN")
    if token:
        return
    from huggingface_hub import HfFolder

    cached_token = HfFolder.get_token()
    if cached_token:
        return
    raise RuntimeError(
        "No Hugging Face authentication found. Run `huggingface-cli login` or set HF_TOKEN before publishing."
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_exists(run_dir: Path, name: str) -> bool:
    return (run_dir / name).exists()


def _build_model_card(runtime: RuntimeConfig, run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    adapter_settings = {
        "standard_lora": asdict(runtime.variant.standard_lora),
        "latent_refiner": asdict(runtime.variant.refiner),
        "latent_refiner_adapter": asdict(runtime.variant.refiner_adapter),
    }
    return "\n".join(
        [
            f"# {metadata.get('run_name', 'recurrent-staged-loras run')}",
            "",
            "## Research artifact notice",
            "This upload is a research artifact from `recurrent-staged-loras`; validate behavior before any downstream usage.",
            "",
            "## Run metadata",
            f"- Base model: `{runtime.variant.base.model_name}`",
            f"- Baseline family: `{metrics.get('baseline_family', runtime.baseline)}`",
            f"- Recurrence mode: `{runtime.variant.refiner.recurrence_mode}`",
            f"- Adapter settings: `{json.dumps(adapter_settings, sort_keys=True)}`",
            f"- Dataset: `{runtime.dataset.get('name')}` split `{runtime.dataset.get('settings', {}).get('split', 'train')}`",
            f"- Training seed: `{runtime.training.seed}`",
            "",
            "## Loading",
            "Primary weights are exported as Hugging Face-compatible safetensors (single-file or sharded with index).",
            "Optional `checkpoint.pt` is included only as a debug/backing artifact.",
        ]
    )


def _build_dataset_card(runtime: RuntimeConfig, partitions_payload: dict[str, Any]) -> str:
    source_name = partitions_payload.get("source_dataset_name", runtime.dataset.get("name"))
    source_split = partitions_payload.get("source_split", runtime.dataset.get("settings", {}).get("split", "train"))
    return "\n".join(
        [
            "# recurrent-staged-loras exported partitions",
            "",
            "## Source",
            f"- Source dataset: `{source_name}`",
            f"- Source split: `{source_split}`",
            "",
            "## Protocol",
            "- Transformation/staging follows repository staged token-mask construction.",
            "- Filtering rules are preserved from `dataset_preprocessing_summary.json`.",
            "- Includes deterministic sample IDs and per-sample fingerprints for reproducibility.",
            "",
            "## Limitations",
            "- These partitions are derived artifacts for experiment reproducibility.",
            "- Publish only if you have rights to share the underlying data.",
            "- Do not upload private/proprietary dataset content without explicit authorization.",
        ]
    )


def _sample_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "sample_id": row.get("sample_id"),
        "answer_text": row.get("answer_text"),
        "answer_text_normalized": row.get("answer_text_normalized"),
        "source_signature": row.get("source_signature"),
        "stage_token_counts": row.get("stage_token_counts"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_datasetdict_payload(partitions_payload: dict[str, Any]) -> DatasetDict:
    train_rows = list(partitions_payload.get("train", []))
    eval_rows = list(partitions_payload.get("eval", []))
    for row in train_rows + eval_rows:
        row["sample_hash"] = _sample_fingerprint(row)
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    )


def publish_run_directory(
    *,
    run_dir: Path,
    runtime: RuntimeConfig,
    publish_cfg: PublishConfig,
    model_repo: str | None = None,
    dataset_repo: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
) -> None:
    _require_hf_auth()
    resolved_model_repo = model_repo or publish_cfg.hub_model_repo
    resolved_dataset_repo = dataset_repo or publish_cfg.hub_dataset_repo
    resolved_private = publish_cfg.private if private is None else private
    resolved_message = commit_message or publish_cfg.commit_message or "Publish recurrent-staged-loras run artifacts"
    api = HfApi()

    if resolved_model_repo:
        api.create_repo(repo_id=resolved_model_repo, private=resolved_private, repo_type="model", exist_ok=True)
        hf_model_dir = run_dir / "hf_model"
        if hf_model_dir.exists():
            shutil.rmtree(hf_model_dir)
        hf_model_dir.mkdir(parents=True, exist_ok=True)

        for name in MODEL_ARTIFACT_CANDIDATES:
            if name == "metrics.json" and not publish_cfg.include_metrics:
                continue
            if name == "answer_eval_diagnostics.json" and not publish_cfg.include_metrics:
                continue
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, hf_model_dir / name)
        serialization = serialize_checkpoint_to_hf_directory(
            run_dir=run_dir,
            output_dir=hf_model_dir,
            runtime_config=runtime.to_serializable_dict(),
            max_shard_size=publish_cfg.max_shard_size,
        )
        validation_result = validate_model_checkpoint(
            base_checkpoint=run_dir / "checkpoint.pt",
            trained_checkpoint=hf_model_dir,
            output_dir=run_dir,
            runtime_config=runtime.to_serializable_dict(),
            validation_cfg=runtime.validation,
        )
        print(
            f"[publish] hf_model_dir='{serialization['model_dir']}' max_shard_size='{serialization['max_shard_size']}' "
            f"validation_report='{validation_result.report_path}' status={'PASS' if validation_result.passed else 'FAIL'}"
        )
        if not validation_result.passed:
            raise RuntimeError(f"Publish validation failed. Report: {validation_result.report_path}")
        if publish_cfg.include_checkpoint and _artifact_exists(run_dir, "checkpoint.pt"):
            shutil.copy2(run_dir / "checkpoint.pt", hf_model_dir / "checkpoint.pt")
        (hf_model_dir / "README.md").write_text(_build_model_card(runtime, run_dir), encoding="utf-8")
        api.upload_folder(
            folder_path=str(hf_model_dir),
            repo_id=resolved_model_repo,
            repo_type="model",
            commit_message=resolved_message,
        )

    if resolved_dataset_repo and publish_cfg.include_dataset_partitions:
        partitions_path = run_dir / "dataset_partitions.json"
        if not partitions_path.exists():
            raise RuntimeError(f"dataset_partitions.json not found at {partitions_path}")
        partitions_payload = _read_json(partitions_path)
        ds_dict = _build_datasetdict_payload(partitions_payload)
        api.create_repo(repo_id=resolved_dataset_repo, private=resolved_private, repo_type="dataset", exist_ok=True)
        ds_dict.push_to_hub(resolved_dataset_repo, private=resolved_private, commit_message=resolved_message)

        with TemporaryDirectory(prefix="hf_dataset_docs_") as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "README.md").write_text(_build_dataset_card(runtime, partitions_payload), encoding="utf-8")
            shutil.copy2(partitions_path, tmp_dir / "dataset_partitions.json")
            preproc_path = run_dir / "dataset_preprocessing_summary.json"
            if preproc_path.exists():
                shutil.copy2(preproc_path, tmp_dir / "dataset_preprocessing_summary.json")
            api.upload_folder(
                folder_path=str(tmp_dir),
                repo_id=resolved_dataset_repo,
                repo_type="dataset",
                commit_message=resolved_message,
            )


def publish_run_directory_from_paths(
    *,
    run_dir: Path,
    model_repo: str | None,
    dataset_repo: str | None,
    private: bool,
    commit_message: str | None,
) -> None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing required config artifact: {cfg_path}")
    from training.config_loader import load_runtime_config_from_raw

    runtime = load_runtime_config_from_raw(_read_json(cfg_path))
    publish_cfg = runtime.publish
    if model_repo is not None:
        publish_cfg.hub_model_repo = model_repo
    if dataset_repo is not None:
        publish_cfg.hub_dataset_repo = dataset_repo
    publish_cfg.private = private
    publish_cfg.commit_message = commit_message
    publish_cfg.enabled = True
    try:
        publish_run_directory(
            run_dir=run_dir,
            runtime=runtime,
            publish_cfg=publish_cfg,
            model_repo=model_repo,
            dataset_repo=dataset_repo,
            private=private,
            commit_message=commit_message,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Hugging Face Hub publish failed: {exc}") from exc
