"""Deterministic answer-evaluation helpers for string and numeric metrics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose
import re
import string

NUMERIC_ABS_TOL = 1e-6
NUMERIC_REL_TOL = 1e-6
NUMERIC_MULTI_VALUE_RULE = "any"


@dataclass(slots=True)
class NumericMatchResult:
    is_match: bool
    predicted_count: int
    target_count: int
    match_count: int
    is_multi_value_target: bool
    skipped: bool = False


def _to_float(token: str) -> float | None:
    text = token.strip()
    if not text:
        return None
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", text):
        left, right = [part.strip() for part in text.split("/", 1)]
        if right in {"0", "+0", "-0"}:
            return None
        try:
            return float(Fraction(int(left), int(right)))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_numeric_values(text: str) -> list[float]:
    if not text:
        return []
    clean = text.replace(",", "")
    pattern = re.compile(r"[-+]?\d+\s*/\s*[-+]?\d+|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
    values: list[float] = []
    for match in pattern.finditer(clean):
        value = _to_float(match.group(0))
        if value is not None:
            values.append(value)
    return values


def normalize_answer_text(text: str, *, semantic_numeric: bool = True) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\$", "", normalized)
    normalized = re.sub(r"\\boxed\{([^}]*)\}", r"\1", normalized)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\frac", "frac")
    normalized = normalized.replace(",", "")
    normalized = re.sub(rf"[{re.escape(string.punctuation.replace('/', ''))}]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not semantic_numeric:
        return normalized

    def _numeric_replacement(match: re.Match[str]) -> str:
        token = match.group(0)
        value = _to_float(token)
        if value is None:
            return token
        if abs(value - round(value)) <= NUMERIC_ABS_TOL:
            return str(int(round(value)))
        return format(value, ".12g")

    normalized = re.sub(r"[-+]?\d+\s*/\s*[-+]?\d+|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", _numeric_replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def numeric_match(
    predicted_text: str,
    target_text: str,
    *,
    abs_tol: float = NUMERIC_ABS_TOL,
    rel_tol: float = NUMERIC_REL_TOL,
    multi_value_rule: str = NUMERIC_MULTI_VALUE_RULE,
) -> NumericMatchResult:
    predicted_values = extract_numeric_values(predicted_text)
    target_values = extract_numeric_values(target_text)
    target_is_multi = len(target_values) > 1
    if not target_values:
        return NumericMatchResult(
            is_match=False,
            predicted_count=len(predicted_values),
            target_count=0,
            match_count=0,
            is_multi_value_target=target_is_multi,
            skipped=True,
        )

    used_pred_indices: set[int] = set()
    matches = 0
    for target in target_values:
        for idx, pred in enumerate(predicted_values):
            if idx in used_pred_indices and multi_value_rule == "all":
                continue
            if isclose(pred, target, rel_tol=rel_tol, abs_tol=abs_tol):
                matches += 1
                used_pred_indices.add(idx)
                break

    if multi_value_rule == "all" and target_is_multi:
        is_match = matches == len(target_values)
    else:
        is_match = matches > 0

    return NumericMatchResult(
        is_match=is_match,
        predicted_count=len(predicted_values),
        target_count=len(target_values),
        match_count=matches,
        is_multi_value_target=target_is_multi,
    )
