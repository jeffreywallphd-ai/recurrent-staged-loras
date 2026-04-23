"""Deterministic answer-evaluation helpers for string, symbolic, and numeric metrics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose
import re
import string

NUMERIC_ABS_TOL = 1e-6
NUMERIC_REL_TOL = 1e-6
NUMERIC_MULTI_VALUE_RULE = "strict_set"
VALID_MULTI_VALUE_RULES = {"strict_set", "subset", "any"}


@dataclass(slots=True)
class NumericMatchResult:
    is_match: bool
    predicted_count: int
    target_count: int
    match_count: int
    is_multi_value_target: bool
    multi_value_status: str
    skipped: bool = False


@dataclass(slots=True)
class SymbolicMatchResult:
    attempted: bool
    parse_success: bool
    is_match: bool


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
    clean = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
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


def _normalize_multi_values(values: list[float], *, abs_tol: float, rel_tol: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    collapsed: list[float] = []
    for value in ordered:
        if collapsed and isclose(value, collapsed[-1], rel_tol=rel_tol, abs_tol=abs_tol):
            continue
        collapsed.append(value)
    return collapsed


def _count_tolerance_matches(pred_values: list[float], target_values: list[float], *, abs_tol: float, rel_tol: float) -> int:
    used_pred_indices: set[int] = set()
    matches = 0
    for target in target_values:
        for idx, pred in enumerate(pred_values):
            if idx in used_pred_indices:
                continue
            if isclose(pred, target, rel_tol=rel_tol, abs_tol=abs_tol):
                matches += 1
                used_pred_indices.add(idx)
                break
    return matches


def numeric_match(
    predicted_text: str,
    target_text: str,
    *,
    abs_tol: float = NUMERIC_ABS_TOL,
    rel_tol: float = NUMERIC_REL_TOL,
    multi_value_rule: str = NUMERIC_MULTI_VALUE_RULE,
) -> NumericMatchResult:
    if multi_value_rule not in VALID_MULTI_VALUE_RULES:
        raise ValueError(f"Unsupported multi-value rule '{multi_value_rule}'. Expected one of {sorted(VALID_MULTI_VALUE_RULES)}")

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
            multi_value_status="skipped_no_target",
            skipped=True,
        )

    match_count = _count_tolerance_matches(predicted_values, target_values, abs_tol=abs_tol, rel_tol=rel_tol)

    if not target_is_multi:
        is_match = match_count > 0
        status = "single_value_match" if is_match else "single_value_unmatched"
    else:
        pred_set = _normalize_multi_values(predicted_values, abs_tol=abs_tol, rel_tol=rel_tol)
        target_set = _normalize_multi_values(target_values, abs_tol=abs_tol, rel_tol=rel_tol)
        strict_set_match = len(pred_set) == len(target_set) and _count_tolerance_matches(pred_set, target_set, abs_tol=abs_tol, rel_tol=rel_tol) == len(target_set)
        any_overlap = _count_tolerance_matches(pred_set, target_set, abs_tol=abs_tol, rel_tol=rel_tol) > 0
        subset_match = bool(pred_set) and _count_tolerance_matches(pred_set, target_set, abs_tol=abs_tol, rel_tol=rel_tol) == len(pred_set)

        if multi_value_rule == "strict_set":
            is_match = strict_set_match
        elif multi_value_rule == "subset":
            is_match = subset_match
        else:  # any
            is_match = any_overlap

        if strict_set_match:
            status = "exact_set_match"
        elif any_overlap:
            status = "partial_overlap"
        else:
            status = "unmatched"

    return NumericMatchResult(
        is_match=is_match,
        predicted_count=len(predicted_values),
        target_count=len(target_values),
        match_count=match_count,
        is_multi_value_target=target_is_multi,
        multi_value_status=status,
    )


def _looks_symbolic_math(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return False
    if re.search(r"[a-zA-Z]", text) and re.search(r"[+\-*/^=()]", text):
        return True
    return bool(re.search(r"(?:\d\s*[+\-*/^]\s*\d)|(?:\()|(?:\bfrac\b)|(?:\\frac)", text))


def symbolic_equivalence_match(predicted_text: str, target_text: str) -> SymbolicMatchResult:
    if not _looks_symbolic_math(predicted_text) and not _looks_symbolic_math(target_text):
        return SymbolicMatchResult(attempted=False, parse_success=False, is_match=False)

    try:
        from sympy import Eq, simplify  # type: ignore
        from sympy.parsing.sympy_parser import parse_expr  # type: ignore
    except ModuleNotFoundError:
        return SymbolicMatchResult(attempted=True, parse_success=False, is_match=False)

    def _prepare(text: str) -> str:
        normalized = text.strip()
        normalized = re.sub(r"\$", "", normalized)
        normalized = re.sub(r"\\boxed\{([^}]*)\}", r"\1", normalized)
        normalized = normalized.replace("^", "**")
        normalized = normalized.replace("\\cdot", "*")
        normalized = normalized.replace("\\times", "*")
        normalized = normalized.replace("\\frac", "frac")
        return normalized.strip()

    def _parse_one(text: str):
        cleaned = _prepare(text)
        if "=" in cleaned:
            left, right = [part.strip() for part in cleaned.split("=", 1)]
            return Eq(parse_expr(left, evaluate=True), parse_expr(right, evaluate=True))
        return parse_expr(cleaned, evaluate=True)

    try:
        pred_expr = _parse_one(predicted_text)
        target_expr = _parse_one(target_text)
    except Exception:
        return SymbolicMatchResult(attempted=True, parse_success=False, is_match=False)

    if isinstance(pred_expr, type(target_expr)) and pred_expr == target_expr:
        return SymbolicMatchResult(attempted=True, parse_success=True, is_match=True)

    try:
        if hasattr(pred_expr, "lhs") and hasattr(pred_expr, "rhs") and hasattr(target_expr, "lhs") and hasattr(target_expr, "rhs"):
            is_match = bool(simplify((pred_expr.lhs - pred_expr.rhs) - (target_expr.lhs - target_expr.rhs)) == 0)
        else:
            is_match = bool(simplify(pred_expr - target_expr) == 0)
    except Exception:
        try:
            is_match = bool(pred_expr.equals(target_expr))
        except Exception:
            is_match = False
    return SymbolicMatchResult(attempted=True, parse_success=True, is_match=is_match)
