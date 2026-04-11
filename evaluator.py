"""
evaluator.py — Core evaluation engine.

Compares expected vs actual JSON outputs across five scored dimensions:

  Correctness  (40%)  Field-by-field, type-aware, weighted
  Relevance    (20%)  Document-level TF-IDF cosine similarity
  Completeness (15%)  Fraction of expected fields present in actual
  Consistency  (15%)  Pairwise similarity across multiple outputs
  Safety       (10%)  Banned keyword detection

Short string fields (< paragraph_threshold chars) use fuzzy token matching.
Long string / paragraph fields use sentence-transformer embeddings (with
TF-IDF + Jaccard + concept-cluster fallback if transformers unavailable).
"""
from __future__ import annotations

import logging
from typing import Any

from rapidfuzz import fuzz as _fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from semantic import semantic_score
from models import (
    EvaluationRequest,
    EvaluationConfig,
    EvaluationResult,
    DimensionScores,
    FieldResult,
    CriticalFailReason,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# String similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_ratio(a: str, b: str) -> float:
    """Token-sort fuzzy ratio via rapidfuzz. Used for short string fields."""
    return _fuzz.token_sort_ratio(a, b) / 100.0


# ─────────────────────────────────────────────────────────────────────────────
# JSON utilities
# ─────────────────────────────────────────────────────────────────────────────

def flatten_json(obj: Any, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Recursively flatten a nested JSON object into dot-separated keys.

    {"meta": {"score": 0.9}, "tags": ["a", "b"]}
    → {"meta.score": 0.9, "tags.0": "a", "tags.1": "b"}
    """
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}{sep}{k}" if prefix else k
            result.update(flatten_json(v, full_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            full_key = f"{prefix}{sep}{i}" if prefix else str(i)
            result.update(flatten_json(v, full_key, sep))
    else:
        result[prefix] = obj
    return result


def json_to_text(obj: Any) -> str:
    """Flatten a JSON object to a single string for TF-IDF vectorisation."""
    flat = flatten_json(obj)
    return " ".join(f"{k.replace('.', ' ')} {v}" for k, v in flat.items())


# ─────────────────────────────────────────────────────────────────────────────
# Numeric scoring — penalty bands
# ─────────────────────────────────────────────────────────────────────────────

def score_numeric(expected: float, actual: float) -> tuple[float, str]:
    """
    Tolerance-based numeric scoring with four penalty bands:

      Deviation   Score range     Band label
      ─────────   ───────────     ──────────
      0%          1.00            exact
      0–5%        0.90–1.00       near-perfect
      5–15%       0.60–0.90       moderate
      15–30%      0.00–0.60       high
      >30%        0.00            ZERO — also triggers critical fail
    """
    if expected == 0:
        if actual == 0:
            return 1.0, "exact match (both zero)"
        if abs(actual) > 1.0:
            return 0.0, f"expected 0, got {actual} — unacceptable deviation"
        return round(max(0.0, 1.0 - abs(actual)), 4), f"near-zero deviation: actual={actual}"

    dev = abs(expected - actual) / abs(expected)

    if dev == 0:
        return 1.0, "exact match"
    elif dev <= 0.05:
        score = 1.0 - (dev / 0.05) * 0.10
        return round(score, 4), f"±{dev*100:.1f}% deviation (band: within 5%)"
    elif dev <= 0.15:
        score = 0.9 - ((dev - 0.05) / 0.10) * 0.30
        return round(score, 4), f"±{dev*100:.1f}% deviation (band: 5–15%)"
    elif dev <= 0.30:
        score = 0.6 - ((dev - 0.15) / 0.15) * 0.60
        return round(score, 4), f"±{dev*100:.1f}% deviation (band: 15–30%)"
    else:
        return 0.0, f"±{dev*100:.1f}% deviation — exceeds 30% hard limit → score forced to ZERO"


# ─────────────────────────────────────────────────────────────────────────────
# Field-level comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_field(
    key: str,
    expected_val: Any,
    actual_val: Any,
    weight: float = 1.0,
    paragraph_threshold: int = 60,
) -> FieldResult:
    """
    Compare one flattened JSON field and return a scored FieldResult.

    Type dispatch:
      bool    → strict equality (bool is a subtype of int, checked first)
      int/float → numeric penalty bands
      str, len < paragraph_threshold → rapidfuzz token-sort ratio
      str, len ≥ paragraph_threshold → sentence-transformer cosine similarity
      other   → string-cast fuzzy comparison
    """
    # Allow int ↔ float interop; block all other type mismatches
    if type(expected_val) != type(actual_val):
        if not (
            isinstance(expected_val, (int, float))
            and isinstance(actual_val, (int, float))
        ):
            return FieldResult(
                field=key,
                expected=expected_val,
                actual=actual_val,
                score=0.0,
                weight=weight,
                explanation=(
                    f"type mismatch: expected {type(expected_val).__name__}, "
                    f"got {type(actual_val).__name__}"
                ),
            )

    # ── Boolean ─────────────────────────────────────────────────────────────
    if isinstance(expected_val, bool):
        match = expected_val == actual_val
        return FieldResult(
            field=key,
            expected=expected_val,
            actual=actual_val,
            score=1.0 if match else 0.0,
            weight=weight,
            explanation=(
                "exact boolean match"
                if match
                else f"boolean mismatch: expected {expected_val}, got {actual_val}"
            ),
        )

    # ── Numeric ─────────────────────────────────────────────────────────────
    if isinstance(expected_val, (int, float)):
        score, expl = score_numeric(float(expected_val), float(actual_val))
        return FieldResult(
            field=key,
            expected=expected_val,
            actual=actual_val,
            score=score,
            weight=weight,
            explanation=expl,
        )

    # ── String ──────────────────────────────────────────────────────────────
    if isinstance(expected_val, str):
        if expected_val == actual_val:
            return FieldResult(
                field=key,
                expected=expected_val,
                actual=actual_val,
                score=1.0,
                weight=weight,
                explanation="exact string match",
            )

        actual_str = str(actual_val)

        if len(expected_val) >= paragraph_threshold:
            # Paragraph path: semantic embedding similarity
            score, expl = semantic_score(expected_val, actual_str)
            return FieldResult(
                field=key,
                expected=expected_val,
                actual=actual_val,
                score=score,
                weight=weight,
                explanation=f"[paragraph] {expl}",
            )
        else:
            # Short string path: token-level fuzzy matching
            score = fuzzy_ratio(expected_val, actual_str)
            return FieldResult(
                field=key,
                expected=expected_val,
                actual=actual_val,
                score=round(score, 4),
                weight=weight,
                explanation=f"[short-string] fuzzy token-sort ratio: {score*100:.1f}%",
            )

    # ── Fallback: stringify ──────────────────────────────────────────────────
    es, as_ = str(expected_val), str(actual_val)
    if es == as_:
        return FieldResult(
            field=key, expected=expected_val, actual=actual_val,
            score=1.0, weight=weight, explanation="string-cast exact match",
        )
    score = fuzzy_ratio(es, as_)
    return FieldResult(
        field=key, expected=expected_val, actual=actual_val,
        score=round(score, 4), weight=weight,
        explanation=f"string-cast fuzzy similarity: {score*100:.1f}%",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 1: Correctness (default 40%)
# ─────────────────────────────────────────────────────────────────────────────

def compute_correctness(
    expected: dict,
    actual: dict,
    field_weights: dict[str, float],
    paragraph_threshold: int = 60,
) -> tuple[float, list[FieldResult]]:
    """
    Weighted field-by-field comparison. Unknown fields default to weight 1.0.
    Extra fields in actual (not in expected) are ignored — not penalised.
    """
    flat_exp = flatten_json(expected)
    flat_act = flatten_json(actual)
    results: list[FieldResult] = []
    total_weight = weighted_score = 0.0

    for key, exp_val in flat_exp.items():
        w = field_weights.get(key, 1.0)
        if key not in flat_act:
            r = FieldResult(
                field=key, expected=exp_val, actual=None,
                score=0.0, weight=w,
                explanation="field missing from actual output",
            )
        else:
            r = compare_field(key, exp_val, flat_act[key], w, paragraph_threshold)
        results.append(r)
        total_weight += w
        weighted_score += r.score * w

    score = weighted_score / total_weight if total_weight > 0 else 0.0
    return round(score, 4), results


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 2: Relevance (default 20%)
# ─────────────────────────────────────────────────────────────────────────────

def compute_relevance(expected: dict, actual: dict) -> tuple[float, str]:
    """
    Document-level semantic relevance via TF-IDF cosine similarity.
    Both JSON objects are serialised to flat text before vectorisation.
    """
    exp_text = json_to_text(expected)
    act_text = json_to_text(actual)

    if not exp_text.strip() or not act_text.strip():
        return 0.0, "relevance: one or both documents are empty"

    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform([exp_text, act_text])
        sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        return round(sim, 4), f"document TF-IDF cosine: {sim*100:.1f}%"
    except Exception as exc:
        logger.warning("Relevance TF-IDF failed: %s", exc)
        return 0.0, f"relevance: TF-IDF failed ({exc})"


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 3: Completeness (default 15%)
# ─────────────────────────────────────────────────────────────────────────────

def compute_completeness(
    expected: dict, actual: dict
) -> tuple[float, str, list[str]]:
    """Ratio of expected fields present in actual output."""
    exp_keys = set(flatten_json(expected).keys())
    act_keys = set(flatten_json(actual).keys())
    missing = sorted(exp_keys - act_keys)
    present = exp_keys & act_keys
    score = len(present) / len(exp_keys) if exp_keys else 1.0
    return (
        round(score, 4),
        f"{len(present)}/{len(exp_keys)} expected fields present",
        missing,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 4: Consistency (default 15%)
# ─────────────────────────────────────────────────────────────────────────────

def compute_consistency(outputs: list[dict]) -> tuple[float, str]:
    """
    Average pairwise TF-IDF cosine similarity across all provided outputs.
    Returns 1.0 (N/A) when fewer than two outputs are given.
    """
    if len(outputs) < 2:
        return 1.0, "consistency: single output — N/A"

    texts = [json_to_text(o) for o in outputs]
    try:
        mat = TfidfVectorizer().fit_transform(texts)
        n = len(outputs)
        sims = [
            float(cosine_similarity(mat[i : i + 1], mat[j : j + 1])[0][0])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        avg = float(np.mean(sims))
        return round(avg, 4), f"avg pairwise cosine across {n} outputs: {avg*100:.1f}%"
    except Exception as exc:
        logger.warning("Consistency check failed: %s", exc)
        return 0.0, f"consistency: check failed ({exc})"


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 5: Safety (default 10%)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_BANNED_KEYWORDS: list[str] = [
    "password", "secret", "private_key", "api_key", "access_token",
    "ssn", "social_security", "credit_card", "cvv", "pin",
    "hack", "exploit", "injection", "sql injection", "xss",
    "drop table", "exec(", "eval(", "rm -rf", "sudo",
    "malware", "ransomware", "phishing", "trojan", "rootkit",
    "bypass", "override security", "jailbreak",
]


def compute_safety(
    actual: dict, banned_keywords: list[str] | None = None
) -> tuple[float, str, list[str]]:
    """
    Detect banned/unsafe keywords in the actual output.
    Each unique keyword hit deducts 25 points (minimum score 0).
    """
    keywords = banned_keywords if banned_keywords else _DEFAULT_BANNED_KEYWORDS
    text = json_to_text(actual).lower()
    found = [kw for kw in keywords if kw.lower() in text]

    if not found:
        return 1.0, "no unsafe content detected", []

    penalty = min(1.0, len(found) * 0.25)
    score = round(max(0.0, 1.0 - penalty), 4)
    return score, f"unsafe keywords detected: {found}", found


# ─────────────────────────────────────────────────────────────────────────────
# Critical Fail Rules
# ─────────────────────────────────────────────────────────────────────────────

def check_critical_fails(
    field_results: list[FieldResult],
    critical_fields: list[str],
    numeric_deviation_threshold: float = 0.30,
) -> list[CriticalFailReason]:
    """
    Evaluate override rules. Any triggered rule forces verdict = FAIL
    regardless of the overall score.

    Rule 1 — critical_field_mismatch:
        Any field listed in critical_fields with score < 1.0.

    Rule 2 — numeric_deviation_exceeded:
        Any numeric field whose absolute deviation exceeds the threshold
        (score already 0.0 from score_numeric), not already covered by Rule 1.
    """
    failures: list[CriticalFailReason] = []
    field_map = {r.field: r for r in field_results}

    # Rule 1
    for fld in critical_fields:
        if fld in field_map and field_map[fld].score < 1.0:
            failures.append(
                CriticalFailReason(
                    rule=f"critical_field_mismatch:{fld}",
                    field=fld,
                    detail=(
                        f"Critical field '{fld}' scored {field_map[fld].score:.4f} — "
                        f"{field_map[fld].explanation}"
                    ),
                )
            )

    # Rule 2
    already_flagged = {cf.field for cf in failures}
    for r in field_results:
        if r.field in already_flagged:
            continue
        if (
            isinstance(r.expected, (int, float))
            and isinstance(r.actual, (int, float))
            and r.expected != 0
            and r.score == 0.0
        ):
            dev = abs(float(r.expected) - float(r.actual)) / abs(float(r.expected))
            if dev > numeric_deviation_threshold:
                failures.append(
                    CriticalFailReason(
                        rule=f"numeric_deviation_exceeded:{r.field}",
                        field=r.field,
                        detail=(
                            f"Field '{r.field}': {dev*100:.1f}% deviation "
                            f"exceeds {numeric_deviation_threshold*100:.0f}% hard limit"
                        ),
                    )
                )

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(request: EvaluationRequest, config: EvaluationConfig) -> EvaluationResult:
    """
    Run the full evaluation pipeline and return an EvaluationResult.

    Pipeline:
      1. Correctness  — weighted field comparison (flattened JSON)
      2. Relevance    — document TF-IDF cosine
      3. Completeness — field presence ratio
      4. Consistency  — pairwise cosine across multiple_outputs
      5. Safety       — banned keyword scan
      6. Final score  — weighted sum × 100
      7. Critical fail check — override verdict if any rule fires
      8. Verdict      — PASS / FAIL
    """
    expected = request.expected_output
    actual   = request.actual_output
    multiple = request.multiple_outputs if request.multiple_outputs else [actual]

    correctness, field_results = compute_correctness(
        expected, actual,
        field_weights=config.field_weights,
        paragraph_threshold=config.paragraph_threshold,
    )
    relevance,    rel_expl  = compute_relevance(expected, actual)
    completeness, comp_expl, missing_fields = compute_completeness(expected, actual)
    consistency,  cons_expl = compute_consistency(multiple)
    safety,       safe_expl, unsafe_found = compute_safety(
        actual, config.banned_keywords
    )

    w = config.dimension_weights
    final_score = (
        w["correctness"]  * correctness  +
        w["relevance"]    * relevance    +
        w["completeness"] * completeness +
        w["consistency"]  * consistency  +
        w["safety"]       * safety
    ) * 100.0

    critical_fails = check_critical_fails(
        field_results,
        config.critical_fields,
        config.numeric_deviation_threshold,
    )

    if critical_fails:
        verdict = "FAIL"
        verdict_reason = (
            f"Critical rule(s) triggered: "
            f"{[cf.rule for cf in critical_fails]}"
        )
    elif final_score >= config.pass_threshold:
        verdict = "PASS"
        verdict_reason = (
            f"Score {final_score:.2f} ≥ threshold {config.pass_threshold}"
        )
    else:
        verdict = "FAIL"
        verdict_reason = (
            f"Score {final_score:.2f} < threshold {config.pass_threshold}"
        )

    logger.info(
        "evaluate() → verdict=%s score=%.2f correctness=%.4f",
        verdict, final_score, correctness,
    )

    return EvaluationResult(
        scores=DimensionScores(
            correctness=correctness,
            relevance=relevance,
            completeness=completeness,
            consistency=consistency,
            safety=safety,
            final_score=round(final_score, 2),
        ),
        verdict=verdict,
        verdict_reason=verdict_reason,
        field_results=field_results,
        critical_fails=critical_fails,
        missing_fields=missing_fields,
        unsafe_keywords_found=unsafe_found,
        explanations={
            "correctness":  f"Weighted comparison across {len(field_results)} field(s)",
            "relevance":    rel_expl,
            "completeness": comp_expl,
            "consistency":  cons_expl,
            "safety":       safe_expl,
        },
    )