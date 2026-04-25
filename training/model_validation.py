"""Checkpoint validation utilities for post-training artifact verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json

import torch


DEFAULT_LORA_KEY_PATTERNS = [
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora",
    # Internal recurrent adapters are modeled as a step-aware adapter bank;
    # include this namespace so stage-specialized runs validate without forcing
    # PEFT-specific key names.
    "adapter_bank.adapters",
]
DEFAULT_RECURRENT_KEY_PATTERNS = [
    "recurrent",
    "recurrence",
    "rnn",
    "recurrent_layer",
    "recurrent_projection",
    "recurrent_gate",
    # In this repository recurrent modules are rooted at `refiner.*`.
    "refiner.",
]


@dataclass(slots=True)
class ModelValidationConfig:
    enabled: bool = True
    blocking: bool = True
    write_json_diff: bool = True
    lora_expected: bool | None = None
    recurrent_expected: bool | None = None
    lora_merged_before_save: bool = False
    lora_key_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_KEY_PATTERNS))
    recurrent_key_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_RECURRENT_KEY_PATTERNS))


@dataclass(slots=True)
class ValidationExpectation:
    lora_expected: bool
    recurrent_expected: bool
    lora_merged_before_save: bool
    lora_key_patterns: list[str]
    recurrent_key_patterns: list[str]


@dataclass(slots=True)
class ModelValidationResult:
    passed: bool
    report_path: Path
    diff_path: Path | None
    missing_required_items: list[str]


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
            return dict(payload)
        for key in ("state_dict", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in nested.items()):
                    return dict(nested)
    raise ValueError("Unable to extract state dict from checkpoint payload")


def _load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file  # type: ignore

    payload = load_file(str(path), device="cpu")
    return _extract_state_dict(payload)


def _list_safetensors_keys(path: Path) -> set[str]:
    from safetensors import safe_open  # type: ignore

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _load_safetensors_with_index(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index_payload = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index_payload.get("weight_map", {})
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"Invalid safetensors index without weight_map at '{path}'")

    state_dict: dict[str, torch.Tensor] = {}
    missing_shards: list[str] = []
    shard_files = sorted(set(str(name) for name in weight_map.values()))
    for shard_name in shard_files:
        shard_path = path.parent / shard_name
        if not shard_path.exists():
            missing_shards.append(shard_name)
            continue
        state_dict.update(_load_safetensors_file(shard_path))

    missing_index_keys = sorted([key for key in weight_map.keys() if key not in state_dict])
    return state_dict, {
        "index_path": str(path),
        "shard_files": shard_files,
        "missing_shard_files": missing_shards,
        "missing_index_keys": missing_index_keys,
        "weight_map": dict(weight_map),
    }


def _safetensors_index_summary(path: Path) -> dict[str, Any]:
    index_payload = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index_payload.get("weight_map", {})
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"Invalid safetensors index without weight_map at '{path}'")

    shard_files = sorted(set(str(name) for name in weight_map.values()))
    missing_shards: list[str] = []
    missing_index_keys: list[str] = []

    shard_key_union: set[str] = set()
    for shard_name in shard_files:
        shard_path = path.parent / shard_name
        if not shard_path.exists():
            missing_shards.append(shard_name)
            continue
        try:
            shard_key_union.update(_list_safetensors_keys(shard_path))
        except Exception:
            # Keep the summary path robust even if a single shard is unreadable.
            missing_index_keys.extend(key for key, name in weight_map.items() if str(name) == shard_name)

    if not missing_index_keys:
        missing_index_keys = sorted([key for key in weight_map.keys() if key not in shard_key_union])
    else:
        missing_index_keys = sorted(set(missing_index_keys))

    return {
        "index_path": str(path),
        "shard_files": shard_files,
        "missing_shard_files": missing_shards,
        "missing_index_keys": missing_index_keys,
        "weight_map": dict(weight_map),
    }


def _load_checkpoint_from_directory(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        state, shard_summary = _load_safetensors_with_index(index_path)
        return state, {
            "serialization_format": "sharded safetensors",
            "hf_model_directory": True,
            "config_present": (path / "config.json").exists(),
            "tokenizer_files": [
                name
                for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "spiece.model")
                if (path / name).exists()
            ],
            "safetensors_index": shard_summary,
            "model_files": ["model.safetensors.index.json", *shard_summary["shard_files"]],
        }

    single_st = path / "model.safetensors"
    if single_st.exists():
        return _load_safetensors_file(single_st), {
            "serialization_format": "single safetensors",
            "hf_model_directory": True,
            "config_present": (path / "config.json").exists(),
            "tokenizer_files": [name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if (path / name).exists()],
            "safetensors_index": None,
            "model_files": ["model.safetensors"],
        }

    for candidate in ("pytorch_model.bin", "checkpoint.pt", "model.pt", "model.bin"):
        candidate_path = path / candidate
        if candidate_path.exists():
            payload = torch.load(candidate_path, map_location="cpu")
            return _extract_state_dict(payload), {
                "serialization_format": "pt/bin",
                "hf_model_directory": True,
                "config_present": (path / "config.json").exists(),
                "tokenizer_files": [name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if (path / name).exists()],
                "safetensors_index": None,
                "model_files": [candidate],
            }

    raise ValueError(f"No supported model file found in directory '{path}'")


def load_checkpoint_state_dict(path_or_obj: Path | str | Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(path_or_obj, Mapping):
        return _extract_state_dict(path_or_obj)

    path = Path(path_or_obj)
    if path.is_dir():
        state, _ = _load_checkpoint_from_directory(path)
        return state

    if path.suffix == ".safetensors":
        return _load_safetensors_file(path)
    if path.name.endswith(".safetensors.index.json"):
        state, _ = _load_safetensors_with_index(path)
        return state

    payload = torch.load(path, map_location="cpu")
    return _extract_state_dict(payload)


def _artifact_summary(path_or_obj: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_obj, Mapping):
        return {"serialization_format": "in-memory mapping", "hf_model_directory": False, "config_present": False, "tokenizer_files": [], "safetensors_index": None, "model_files": []}
    path = Path(path_or_obj)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            shard_summary = _safetensors_index_summary(index_path)
            return {
                "serialization_format": "sharded safetensors",
                "hf_model_directory": True,
                "config_present": (path / "config.json").exists(),
                "tokenizer_files": [
                    name
                    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "spiece.model")
                    if (path / name).exists()
                ],
                "safetensors_index": shard_summary,
                "model_files": ["model.safetensors.index.json", *shard_summary["shard_files"]],
            }
        single_st = path / "model.safetensors"
        if single_st.exists():
            return {
                "serialization_format": "single safetensors",
                "hf_model_directory": True,
                "config_present": (path / "config.json").exists(),
                "tokenizer_files": [
                    name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if (path / name).exists()
                ],
                "safetensors_index": None,
                "model_files": ["model.safetensors"],
            }
        for candidate in ("pytorch_model.bin", "checkpoint.pt", "model.pt", "model.bin"):
            candidate_path = path / candidate
            if candidate_path.exists():
                return {
                    "serialization_format": "pt/bin",
                    "hf_model_directory": True,
                    "config_present": (path / "config.json").exists(),
                    "tokenizer_files": [
                        name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if (path / name).exists()
                    ],
                    "safetensors_index": None,
                    "model_files": [candidate],
                }
        raise ValueError(f"No supported model file found in directory '{path}'")
    if path.suffix == ".safetensors":
        return {"serialization_format": "single safetensors", "hf_model_directory": False, "config_present": False, "tokenizer_files": [], "safetensors_index": None, "model_files": [path.name]}
    if path.name.endswith(".safetensors.index.json"):
        shard_summary = _safetensors_index_summary(path)
        return {"serialization_format": "sharded safetensors", "hf_model_directory": False, "config_present": False, "tokenizer_files": [], "safetensors_index": shard_summary, "model_files": [path.name, *shard_summary["shard_files"]]}
    return {"serialization_format": "pt/bin", "hf_model_directory": False, "config_present": False, "tokenizer_files": [], "safetensors_index": None, "model_files": [path.name]}


def _normalize_key(key: str) -> str:
    normalized = key
    while normalized.startswith("module."):
        normalized = normalized[len("module.") :]
    while normalized.startswith("model."):
        normalized = normalized[len("model.") :]
    return normalized


def _normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        normalized[_normalize_key(str(key))] = tensor
    return normalized


def _matches_any_pattern(key: str, patterns: list[str]) -> bool:
    lowered = key.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _numel(tensor: torch.Tensor) -> int:
    return int(tensor.numel())


def _lookup_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _default_expectation(*, runtime_config: Mapping[str, Any], validation_cfg: ModelValidationConfig) -> ValidationExpectation:
    refiner_cfg = runtime_config.get("variant", {}).get("refiner", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    standard_lora_cfg = runtime_config.get("variant", {}).get("standard_lora", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    refiner_adapter_cfg = runtime_config.get("variant", {}).get("refiner_adapter", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}

    inferred_lora = bool(standard_lora_cfg.get("enabled", False) or refiner_adapter_cfg.get("enabled", False))
    inferred_recurrent = bool(refiner_cfg.get("enabled", False))
    return ValidationExpectation(
        lora_expected=inferred_lora if validation_cfg.lora_expected is None else bool(validation_cfg.lora_expected),
        recurrent_expected=inferred_recurrent if validation_cfg.recurrent_expected is None else bool(validation_cfg.recurrent_expected),
        lora_merged_before_save=bool(validation_cfg.lora_merged_before_save),
        lora_key_patterns=list(validation_cfg.lora_key_patterns),
        recurrent_key_patterns=list(validation_cfg.recurrent_key_patterns),
    )


def _runtime_any_lora_enabled(runtime_config: Mapping[str, Any]) -> bool:
    variant_cfg = runtime_config.get("variant", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    standard_lora_cfg = variant_cfg.get("standard_lora", {}) if isinstance(variant_cfg, Mapping) else {}
    refiner_adapter_cfg = variant_cfg.get("refiner_adapter", {}) if isinstance(variant_cfg, Mapping) else {}
    return bool(standard_lora_cfg.get("enabled", False) or refiner_adapter_cfg.get("enabled", False))


def _merged_lora_metadata_present(trained_path: Path | str | Mapping[str, Any], runtime_config: Mapping[str, Any], expectation: ValidationExpectation) -> bool:
    if expectation.lora_merged_before_save:
        return True
    if isinstance(trained_path, Mapping):
        return False
    path = Path(trained_path)
    if path.is_dir():
        for metadata_name in ("serialization_metadata.json", "training_metadata.json", "metadata.json"):
            candidate = path / metadata_name
            if candidate.exists():
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if bool(payload.get("lora_merged", False)):
                    return True
        cfg_path = path / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                if bool(cfg.get("lora_merged", False)):
                    return True
            except Exception:
                pass
    return bool(_lookup_nested(runtime_config, "validation.lora_merged_before_save"))


def validate_model_checkpoint(*,
    base_checkpoint: Path | str | Mapping[str, Any],
    trained_checkpoint: Path | str | Mapping[str, Any],
    output_dir: Path | str,
    runtime_config: Mapping[str, Any],
    validation_cfg: ModelValidationConfig,
) -> ModelValidationResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_state = _normalize_state_dict(load_checkpoint_state_dict(base_checkpoint))
    trained_state = _normalize_state_dict(load_checkpoint_state_dict(trained_checkpoint))
    expectation = _default_expectation(runtime_config=runtime_config, validation_cfg=validation_cfg)

    base_summary = _artifact_summary(base_checkpoint)
    trained_summary = _artifact_summary(trained_checkpoint)

    base_keys = set(base_state)
    trained_keys = set(trained_state)
    added_keys = sorted(trained_keys - base_keys)
    removed_keys = sorted(base_keys - trained_keys)
    shared_keys = sorted(base_keys & trained_keys)

    shape_changes: list[dict[str, Any]] = []
    unchanged_shared_count = 0
    for key in shared_keys:
        base_shape = list(base_state[key].shape)
        trained_shape = list(trained_state[key].shape)
        if base_shape != trained_shape:
            shape_changes.append({"key": key, "base_shape": base_shape, "trained_shape": trained_shape})
        else:
            unchanged_shared_count += 1

    lora_keys = sorted([key for key in trained_keys if _matches_any_pattern(key, expectation.lora_key_patterns)])
    recurrent_keys = sorted([key for key in trained_keys if _matches_any_pattern(key, expectation.recurrent_key_patterns)])

    config_comparison = {
        "variant.refiner.enabled": {
            "expected": expectation.recurrent_expected,
            "actual": _lookup_nested(runtime_config, "variant.refiner.enabled"),
            "matches": _lookup_nested(runtime_config, "variant.refiner.enabled") == expectation.recurrent_expected,
        },
        "variant.any_lora.enabled": {
            "expected": expectation.lora_expected,
            "actual": _runtime_any_lora_enabled(runtime_config),
            "matches": _runtime_any_lora_enabled(runtime_config) == expectation.lora_expected,
        },
        "variant.base.architecture_type": {
            "expected": _lookup_nested(runtime_config, "variant.base.architecture_type"),
            "actual": _lookup_nested(runtime_config, "variant.base.architecture_type"),
            "matches": _lookup_nested(runtime_config, "variant.base.architecture_type") is not None,
        },
        "variant.refiner.num_steps": {
            "expected": _lookup_nested(runtime_config, "variant.refiner.num_steps"),
            "actual": _lookup_nested(runtime_config, "variant.refiner.num_steps"),
            "matches": _lookup_nested(runtime_config, "variant.refiner.num_steps") is not None,
        },
    }

    adapter_files: list[str] = []
    merged_lora_metadata = _merged_lora_metadata_present(trained_checkpoint, runtime_config, expectation)
    if not isinstance(trained_checkpoint, Mapping):
        trained_path = Path(trained_checkpoint)
        if trained_path.is_dir():
            adapter_files = [
                name
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.safetensors.index.json",
                )
                if (trained_path / name).exists()
            ]

    missing_required_items: list[str] = []
    if expectation.lora_expected and not lora_keys and not merged_lora_metadata and not adapter_files:
        missing_required_items.append("Expected LoRA keys/files were not found, and no merged-LoRA metadata was provided")
    if expectation.recurrent_expected and not recurrent_keys:
        missing_required_items.append("Expected recurrent-layer keys were not found in the trained checkpoint")
    for key, cmp in config_comparison.items():
        if not bool(cmp["matches"]):
            missing_required_items.append(f"Config expectation mismatch for '{key}'")

    if trained_summary.get("hf_model_directory"):
        if not trained_summary.get("config_present"):
            missing_required_items.append("HF compatibility check failed: missing config.json")
        idx = trained_summary.get("safetensors_index")
        if isinstance(idx, Mapping):
            if idx.get("missing_shard_files"):
                missing_required_items.append("HF compatibility check failed: safetensors index references missing shard files")
            if idx.get("missing_index_keys"):
                missing_required_items.append("HF compatibility check failed: index references keys missing from shard files")

    parameter_summary = {
        "base_only_parameters": int(sum(_numel(base_state[k]) for k in removed_keys)),
        "trained_only_parameters": int(sum(_numel(trained_state[k]) for k in added_keys)),
        "shared_parameters": int(sum(_numel(trained_state[k]) for k in shared_keys)),
        "lora_parameters": int(sum(_numel(trained_state[k]) for k in lora_keys)),
        "recurrent_parameters": int(sum(_numel(trained_state[k]) for k in recurrent_keys)),
    }

    passed = len(missing_required_items) == 0
    report_path = out_dir / "model_validation_report.md"
    diff_path = out_dir / "model_validation_diff.json" if validation_cfg.write_json_diff else None

    shard_summary = trained_summary.get("safetensors_index") if isinstance(trained_summary.get("safetensors_index"), Mapping) else {}
    summary_lines = [
        "# Model Validation Report",
        "",
        "## Validation Summary",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Base checkpoint path: `{base_checkpoint}`",
        f"- Trained checkpoint path: `{trained_checkpoint}`",
        f"- Run output path: `{out_dir}`",
        "",
        "## Serialization Format",
        f"- Base: {base_summary.get('serialization_format')}",
        f"- Trained: {trained_summary.get('serialization_format')}",
        f"- Trained is HF model directory: {'yes' if trained_summary.get('hf_model_directory') else 'no'}",
        "",
        "## Shard Summary",
        f"- Shard count: {len(shard_summary.get('shard_files', [])) if shard_summary else 0}",
        f"- Total tensor count: {len(trained_keys)}",
        f"- Index file path: `{shard_summary.get('index_path') if shard_summary else ''}`",
        f"- Missing shard files: `{shard_summary.get('missing_shard_files', []) if shard_summary else []}`",
        "",
        "## HF Compatibility Check",
        f"- config.json present: {'yes' if trained_summary.get('config_present') else 'no'}",
        f"- tokenizer files present: `{trained_summary.get('tokenizer_files', [])}`",
        f"- safetensors index valid: {'yes' if not shard_summary or (not shard_summary.get('missing_shard_files') and not shard_summary.get('missing_index_keys')) else 'no'}",
        f"- model files detected: `{trained_summary.get('model_files', [])}`",
        "",
        "## LoRA Handling",
        f"- unmerged adapter keys found: {len(lora_keys)}",
        f"- adapter files found: `{adapter_files}`",
        f"- merged LoRA metadata found: {'yes' if merged_lora_metadata else 'no'}",
        "",
        "## Recurrent Layer Validation",
        f"- expected recurrent keys/config found: {'yes' if (not expectation.recurrent_expected or bool(recurrent_keys)) else 'no'}",
        "",
        "## Expected Additions",
        f"- LoRA expected: {'yes' if expectation.lora_expected else 'no'}",
        f"- Recurrent expected: {'yes' if expectation.recurrent_expected else 'no'}",
        f"- LoRA merged before save: {'yes' if expectation.lora_merged_before_save else 'no'}",
        f"- LoRA key patterns: `{expectation.lora_key_patterns}`",
        f"- Recurrent key patterns: `{expectation.recurrent_key_patterns}`",
        "",
        "## Diff View",
        f"- Added keys (`+`): {len(added_keys)}",
    ]
    summary_lines.extend([f"  - `+ {key}`" for key in added_keys])
    summary_lines.append(f"- Missing/removed keys (`-`): {len(removed_keys)}")
    summary_lines.extend([f"  - `- {key}`" for key in removed_keys])
    summary_lines.append(f"- Shape changes (`~`): {len(shape_changes)}")
    summary_lines.extend([f"  - `~ {item['key']}: {item['base_shape']} -> {item['trained_shape']}`" for item in shape_changes])
    summary_lines.append(f"- Unchanged shared keys (summarized): {unchanged_shared_count}")
    summary_lines.extend([
        "",
        "## Config Comparison",
    ])
    for key, cmp in config_comparison.items():
        summary_lines.append(f"- `{key}` expected=`{cmp['expected']}` actual=`{cmp['actual']}` match=`{cmp['matches']}`")

    summary_lines.extend([
        "",
        "## Parameter Count Summary",
        f"- Base-only parameters: {parameter_summary['base_only_parameters']}",
        f"- Trained-only parameters: {parameter_summary['trained_only_parameters']}",
        f"- Shared parameters: {parameter_summary['shared_parameters']}",
        f"- LoRA parameters: {parameter_summary['lora_parameters']}",
        f"- Recurrent/additional-layer parameters: {parameter_summary['recurrent_parameters']}",
        "",
        "## Recommended Action",
        "- Safe to continue." if passed else "- Stop and inspect missing items before considering this model ready.",
    ])
    if missing_required_items:
        summary_lines.append("- Missing required items:")
        summary_lines.extend([f"  - {item}" for item in missing_required_items])

    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if diff_path is not None:
        diff_payload = {
            "passed": passed,
            "base_checkpoint": str(base_checkpoint),
            "trained_checkpoint": str(trained_checkpoint),
            "output_dir": str(out_dir),
            "expected": asdict(expectation),
            "missing_required_items": missing_required_items,
            "serialization": {
                "base": base_summary,
                "trained": trained_summary,
            },
            "added_keys": added_keys,
            "removed_keys": removed_keys,
            "shape_changes": shape_changes,
            "unchanged_shared_key_count": unchanged_shared_count,
            "lora_keys": lora_keys,
            "recurrent_keys": recurrent_keys,
            "adapter_files": adapter_files,
            "merged_lora_metadata": merged_lora_metadata,
            "config_comparison": config_comparison,
            "parameter_summary": parameter_summary,
        }
        diff_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    return ModelValidationResult(passed=passed, report_path=report_path, diff_path=diff_path, missing_required_items=missing_required_items)
