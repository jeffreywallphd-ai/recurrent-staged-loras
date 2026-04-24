"""Checkpoint validation utilities for post-training artifact verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json

import torch


DEFAULT_LORA_KEY_PATTERNS = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora"]
DEFAULT_RECURRENT_KEY_PATTERNS = ["recurrent", "recurrence", "rnn", "recurrent_layer", "recurrent_projection", "recurrent_gate"]


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


def load_checkpoint_state_dict(path_or_obj: Path | str | Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(path_or_obj, Mapping):
        return _extract_state_dict(path_or_obj)

    path = Path(path_or_obj)
    if path.is_dir():
        for candidate in ("pytorch_model.bin", "model.safetensors"):
            candidate_path = path / candidate
            if not candidate_path.exists():
                continue
            if candidate.endswith(".safetensors"):
                from safetensors.torch import load_file  # type: ignore

                payload = load_file(str(candidate_path), device="cpu")
                return _extract_state_dict(payload)
            payload = torch.load(candidate_path, map_location="cpu")
            return _extract_state_dict(payload)
        raise ValueError(f"No supported model file found in directory '{path}'")

    payload = torch.load(path, map_location="cpu")
    return _extract_state_dict(payload)


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
    model_cfg = runtime_config.get("variant", {}).get("base", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    refiner_cfg = runtime_config.get("variant", {}).get("refiner", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    standard_lora_cfg = runtime_config.get("variant", {}).get("standard_lora", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    refiner_adapter_cfg = runtime_config.get("variant", {}).get("refiner_adapter", {}) if isinstance(runtime_config.get("variant"), Mapping) else {}
    del model_cfg

    inferred_lora = bool(standard_lora_cfg.get("enabled", False) or refiner_adapter_cfg.get("enabled", False))
    inferred_recurrent = bool(refiner_cfg.get("enabled", False))
    return ValidationExpectation(
        lora_expected=inferred_lora if validation_cfg.lora_expected is None else bool(validation_cfg.lora_expected),
        recurrent_expected=inferred_recurrent if validation_cfg.recurrent_expected is None else bool(validation_cfg.recurrent_expected),
        lora_merged_before_save=bool(validation_cfg.lora_merged_before_save),
        lora_key_patterns=list(validation_cfg.lora_key_patterns),
        recurrent_key_patterns=list(validation_cfg.recurrent_key_patterns),
    )


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

    config_expectations = {
        "variant.refiner.enabled": expectation.recurrent_expected,
        "variant.standard_lora.enabled": expectation.lora_expected,
        "variant.base.architecture_type": _lookup_nested(runtime_config, "variant.base.architecture_type"),
        "variant.refiner.num_steps": _lookup_nested(runtime_config, "variant.refiner.num_steps"),
    }
    config_comparison = {
        dotted: {
            "expected": expected,
            "actual": _lookup_nested(runtime_config, dotted),
            "matches": _lookup_nested(runtime_config, dotted) == expected if expected is not None else _lookup_nested(runtime_config, dotted) is not None,
        }
        for dotted, expected in config_expectations.items()
    }

    missing_required_items: list[str] = []
    if expectation.lora_expected and not expectation.lora_merged_before_save and not lora_keys:
        missing_required_items.append("Expected LoRA keys were not found in the trained checkpoint")
    if expectation.recurrent_expected and not recurrent_keys:
        missing_required_items.append("Expected recurrent-layer keys were not found in the trained checkpoint")
    for key, cmp in config_comparison.items():
        if not bool(cmp["matches"]):
            missing_required_items.append(f"Config expectation mismatch for '{key}'")

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

    summary_lines = [
        "# Model Validation Report",
        "",
        "## Validation Summary",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Base checkpoint path: `{base_checkpoint}`",
        f"- Trained checkpoint path: `{trained_checkpoint}`",
        f"- Run output path: `{out_dir}`",
        "",
        "## Expected Additions",
        f"- LoRA expected: {'yes' if expectation.lora_expected else 'no'}",
        f"- Recurrent expected: {'yes' if expectation.recurrent_expected else 'no'}",
        f"- LoRA merged before save: {'yes' if expectation.lora_merged_before_save else 'no'}",
        f"- LoRA key patterns: `{expectation.lora_key_patterns}`",
        f"- Recurrent key patterns: `{expectation.recurrent_key_patterns}`",
        "",
        "## Checkpoint Key Diff",
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
        "## Detected LoRA Keys",
        f"- Count: {len(lora_keys)}",
    ])
    summary_lines.extend([f"  - `{key}`" for key in lora_keys])
    summary_lines.extend([
        "",
        "## Detected Recurrent Keys",
        f"- Count: {len(recurrent_keys)}",
    ])
    summary_lines.extend([f"  - `{key}`" for key in recurrent_keys])

    summary_lines.extend(["", "## Config Comparison"])
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
            "added_keys": added_keys,
            "removed_keys": removed_keys,
            "shape_changes": shape_changes,
            "unchanged_shared_key_count": unchanged_shared_count,
            "lora_keys": lora_keys,
            "recurrent_keys": recurrent_keys,
            "config_comparison": config_comparison,
            "parameter_summary": parameter_summary,
        }
        diff_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    return ModelValidationResult(passed=passed, report_path=report_path, diff_path=diff_path, missing_required_items=missing_required_items)
