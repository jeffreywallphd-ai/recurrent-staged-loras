"""Model serialization helpers for Hugging Face-compatible sharded safetensors output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import shutil

import torch
from huggingface_hub import save_torch_state_dict

from training.model_validation import load_checkpoint_state_dict

MODEL_METADATA_FILES = [
    "metadata.json",
    "metrics.json",
    "answer_eval_diagnostics.json",
    "dataset_preprocessing_summary.json",
    "config.json",
]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _write_model_config(*, output_dir: Path, runtime_config: Mapping[str, Any]) -> None:
    base = runtime_config.get("variant", {}).get("base", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    payload: dict[str, Any] = {
        "model_name_or_path": base.get("model_name"),
        "architecture_type": base.get("architecture_type"),
        "max_seq_length": base.get("max_seq_length"),
        "trust_remote_code": base.get("trust_remote_code", False),
        "recurrent_staged_loras": runtime_config,
    }
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_serialization_metadata(*, output_dir: Path, max_shard_size: str) -> None:
    (output_dir / "serialization_metadata.json").write_text(
        json.dumps(
            {
                "format": "safetensors",
                "sharded": (output_dir / "model.safetensors.index.json").exists(),
                "max_shard_size": max_shard_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _copy_adapter_files(run_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    adapter_candidates = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.safetensors.index.json",
    ]
    for name in adapter_candidates:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    if (run_dir / "adapter_model.safetensors.index.json").exists():
        idx = json.loads((run_dir / "adapter_model.safetensors.index.json").read_text(encoding="utf-8"))
        for filename in sorted(set(idx.get("weight_map", {}).values())):
            src = run_dir / str(filename)
            if src.exists():
                shutil.copy2(src, output_dir / str(filename))
                copied.append(str(filename))
    return sorted(set(copied))


def _copy_tokenizer_if_available(*, run_dir: Path, output_dir: Path, runtime_config: Mapping[str, Any]) -> list[str]:
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    if copied:
        return sorted(copied)

    try:
        from transformers import AutoTokenizer  # type: ignore

        variant = runtime_config.get("variant", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
        base = variant.get("base", {}) if isinstance(variant, Mapping) else {}
        tok_name = base.get("tokenizer_name") or base.get("model_name")
        if tok_name:
            tokenizer = AutoTokenizer.from_pretrained(str(tok_name), local_files_only=True)
            tokenizer.save_pretrained(str(output_dir))
            for name in TOKENIZER_FILES:
                if (output_dir / name).exists():
                    copied.append(name)
    except Exception:
        return sorted(set(copied))
    return sorted(set(copied))


def serialize_checkpoint_to_hf_directory(
    *,
    run_dir: Path,
    output_dir: Path,
    runtime_config: Mapping[str, Any],
    max_shard_size: str = "4GB",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing checkpoint for HF serialization: {checkpoint_path}")

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    save_torch_state_dict(
        state_dict=state_dict,
        save_directory=str(output_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
        filename_pattern="model{suffix}.safetensors",
    )
    _write_model_config(output_dir=output_dir, runtime_config=runtime_config)
    _write_serialization_metadata(output_dir=output_dir, max_shard_size=max_shard_size)

    for name in MODEL_METADATA_FILES:
        if name == "config.json":
            continue
        _copy_if_exists(run_dir / name, output_dir / name)

    tokenizer_files = _copy_tokenizer_if_available(run_dir=run_dir, output_dir=output_dir, runtime_config=runtime_config)
    adapter_files = _copy_adapter_files(run_dir=run_dir, output_dir=output_dir)

    return {
        "model_dir": output_dir,
        "max_shard_size": max_shard_size,
        "tokenizer_files": tokenizer_files,
        "adapter_files": adapter_files,
        "sharded": (output_dir / "model.safetensors.index.json").exists(),
    }
