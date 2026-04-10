"""
Prompt Evaluation Engine - Core Evaluator
Deterministic, explainable, production-ready AI output scoring.

Paragraph/semantic support: string fields longer than `paragraph_threshold`
chars are scored via TF-IDF cosine (bigrams) instead of fuzzy char-matching.

Production upgrade path for even better semantic accuracy:
    pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    _ST = SentenceTransformer("all-MiniLM-L6-v2")
    def semantic_score(a, b): e = _ST.encode([a,b]); return cosine_similarity([e[0]],[e[1]])[0][0]
"""
from __future__ import annotations
import re, math
from typing import Any

# fuzzy_ratio: used for SHORT strings only (labels, categories, codes)
try:
    from rapidfuzz import fuzz as _fuzz
    def fuzzy_ratio(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0
except ImportError:
    import difflib
    def fuzzy_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

from semantic import semantic_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from semantic import semantic_score          # multi-signal paragraph scorer
from models import (
    EvaluationRequest, EvaluationConfig, EvaluationResult,
    DimensionScores, FieldResult, CriticalFailReason
)


# ─────────────────────────────────────────────
# Semantic Paragraph Scorer
# ─────────────────────────────────────────────

def semantic_score(expected_text: str, actual_text: str) -> tuple[float, str]:
    """
    Compute semantic similarity between two text passages using TF-IDF cosine.
    Used for paragraph/long-text fields where fuzzy char-matching fails.

    Uses unigram+bigram TF-IDF with sublinear_tf to capture phrase-level meaning.
    Score is reliable when both texts share topical vocabulary even with
    completely different surface wording.

    Production upgrade: swap body with sentence-transformers for true semantic
    embeddings (handles synonyms, paraphrases across different vocabulary).
    """
    if not expected_text.strip() or not actual_text.strip():
        return 0.0, "semantic: empty text"
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        mat = vec.fit_transform([expected_text, actual_text])
        sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        return round(sim, 4), f"semantic/TF-IDF-bigram cosine: {sim*100:.1f}%"
    except Exception as e:
        s = fuzzy_ratio(expected_text, actual_text)
        return round(s, 4), f"semantic fallback (fuzzy): {s*100:.1f}% — TF-IDF error: {e}"


# ─────────────────────────────────────────────
# JSON Utilities
# ─────────────────────────────────────────────

def flatten_json(obj: Any, prefix: str = "", sep: str = ".") -> dict:
    result = {}
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
    flat = flatten_json(obj)
    return " ".join(f"{k.replace('.', ' ')} {v}" for k, v in flat.items())


# ─────────────────────────────────────────────
# Numeric Scoring with Penalty Bands
# ─────────────────────────────────────────────

def score_numeric(expected: float, actual: float) -> tuple[float, str]:
    """
    Penalty bands:
      0–5%   → 0.90–1.00   near-perfect
      5–15%  → 0.60–0.90   moderate
      15–30% → 0.00–0.60   high
      >30%   → 0.00        ZERO (critical)
    """
    if expected == 0:
        if actual == 0:
            return 1.0, "exact match (both zero)"
        if abs(actual) > 1.0:
            return 0.0, f"expected 0, got {actual} — large absolute deviation"
        return max(0.0, 1.0 - abs(actual)), f"small deviation from zero: {actual}"

    dev = abs(expected - actual) / abs(expected)
    if dev == 0:
        return 1.0, "exact match"
    elif dev <= 0.05:
        return round(1.0 - (dev / 0.05) * 0.1, 4), f"±{dev*100:.1f}% deviation (within 5%)"
    elif dev <= 0.15:
        return round(0.9 - ((dev - 0.05) / 0.10) * 0.30, 4), f"±{dev*100:.1f}% deviation (moderate, 5–15%)"
    elif dev <= 0.30:
        return round(0.6 - ((dev - 0.15) / 0.15) * 0.60, 4), f"±{dev*100:.1f}% deviation (high, 15–30%)"
    else:
        return 0.0, f"±{dev*100:.1f}% deviation — exceeds 30% threshold → ZERO"


# ─────────────────────────────────────────────
# Field-Level Comparison  ← CHANGED: paragraph routing added
# ─────────────────────────────────────────────

def compare_field(key: str, expected_val: Any, actual_val: Any,
                  weight: float = 1.0,
                  paragraph_threshold: int = 60) -> FieldResult:
    """
    Compare a single flattened field.

    String routing logic:
      len(expected) < paragraph_threshold  → fuzzy_ratio   (token-level)
      len(expected) >= paragraph_threshold → semantic_score (TF-IDF cosine)

    paragraph_threshold default = 60 chars (~1 short sentence boundary).
    Set lower (e.g. 30) to be more aggressive with semantic scoring.
    """
    # Type mismatch (allow int/float interop)
    if type(expected_val) != type(actual_val):
        if not (isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float))):
            return FieldResult(
                field=key, expected=expected_val, actual=actual_val, score=0.0,
                weight=weight,
                explanation=f"type mismatch: expected {type(expected_val).__name__}, got {type(actual_val).__name__}"
            )

    # Boolean (check before numeric — bool is int subtype)
    if isinstance(expected_val, bool):
        match = expected_val == actual_val
        return FieldResult(
            field=key, expected=expected_val, actual=actual_val,
            score=1.0 if match else 0.0, weight=weight,
            explanation="exact boolean match" if match
                        else f"boolean mismatch: expected {expected_val}, got {actual_val}"
        )

    # Numeric
    if isinstance(expected_val, (int, float)):
        s, expl = score_numeric(float(expected_val), float(actual_val))
        return FieldResult(field=key, expected=expected_val, actual=actual_val,
                           score=s, weight=weight, explanation=expl)

    # ── String — route by length ─────────────────────────────────────────
    if isinstance(expected_val, str):
        if expected_val == actual_val:
            return FieldResult(field=key, expected=expected_val, actual=actual_val,
                               score=1.0, weight=weight, explanation="exact string match")

        if len(expected_val) >= paragraph_threshold:
            # PARAGRAPH PATH: semantic TF-IDF cosine
            s, expl = semantic_score(expected_val, str(actual_val))
            return FieldResult(field=key, expected=expected_val, actual=actual_val,
                               score=s, weight=weight,
                               explanation=f"[paragraph] {expl}")
        else:
            # SHORT STRING PATH: fuzzy character/token matching
            s = fuzzy_ratio(expected_val, str(actual_val))
            return FieldResult(field=key, expected=expected_val, actual=actual_val,
                               score=round(s, 4), weight=weight,
                               explanation=f"[short-string] fuzzy similarity: {s*100:.1f}%")

    # Fallback: stringify
    es, as_ = str(expected_val), str(actual_val)
    if es == as_:
        return FieldResult(field=key, expected=expected_val, actual=actual_val,
                           score=1.0, weight=weight, explanation="string-cast exact match")
    s = fuzzy_ratio(es, as_)
    return FieldResult(field=key, expected=expected_val, actual=actual_val,
                       score=round(s, 4), weight=weight,
                       explanation=f"string-cast fuzzy similarity: {s*100:.1f}%")


# ─────────────────────────────────────────────
# Dimension 1: Correctness (40%)
# ─────────────────────────────────────────────

def compute_correctness(expected: dict, actual: dict,
                         field_weights: dict,
                         paragraph_threshold: int = 60) -> tuple[float, list[FieldResult]]:
    flat_exp = flatten_json(expected)
    flat_act = flatten_json(actual)
    results: list[FieldResult] = []
    total_w = weighted_s = 0.0

    for key, exp_val in flat_exp.items():
        w = field_weights.get(key, 1.0)
        if key not in flat_act:
            r = FieldResult(field=key, expected=exp_val, actual=None,
                            score=0.0, weight=w, explanation="field missing in actual output")
        else:
            r = compare_field(key, exp_val, flat_act[key], w, paragraph_threshold)
        results.append(r)
        total_w += w
        weighted_s += r.score * w

    score = weighted_s / total_w if total_w > 0 else 0.0
    return round(score, 4), results


# ─────────────────────────────────────────────
# Dimension 2: Relevance (20%)
# ─────────────────────────────────────────────

def compute_relevance(expected: dict, actual: dict) -> tuple[float, str]:
    exp_text = json_to_text(expected)
    act_text = json_to_text(actual)
    if not exp_text.strip() or not act_text.strip():
        return 0.0, "empty document(s)"
    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform([exp_text, act_text])
        sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        return round(sim, 4), f"TF-IDF cosine similarity: {sim*100:.1f}%"
    except Exception as e:
        return 0.0, f"TF-IDF failed: {e}"


# ─────────────────────────────────────────────
# Dimension 3: Completeness (15%)
# ─────────────────────────────────────────────

def compute_completeness(expected: dict, actual: dict) -> tuple[float, str, list[str]]:
    exp_keys = set(flatten_json(expected).keys())
    act_keys = set(flatten_json(actual).keys())
    missing = sorted(exp_keys - act_keys)
    present = exp_keys & act_keys
    score = len(present) / len(exp_keys) if exp_keys else 1.0
    return round(score, 4), f"{len(present)}/{len(exp_keys)} expected fields present", missing


# ─────────────────────────────────────────────
# Dimension 4: Consistency (15%)
# ─────────────────────────────────────────────

def compute_consistency(outputs: list[dict]) -> tuple[float, str]:
    if not outputs or len(outputs) < 2:
        return 1.0, "single output — consistency N/A"
    texts = [json_to_text(o) for o in outputs]
    try:
        mat = TfidfVectorizer().fit_transform(texts)
        n = len(outputs)
        sims = [float(cosine_similarity(mat[i:i+1], mat[j:j+1])[0][0])
                for i in range(n) for j in range(i+1, n)]
        avg = float(np.mean(sims))
        return round(avg, 4), f"avg pairwise cosine across {n} outputs: {avg*100:.1f}%"
    except Exception as e:
        return 0.0, f"consistency check failed: {e}"


# ─────────────────────────────────────────────
# Dimension 5: Safety (10%)
# ─────────────────────────────────────────────

DEFAULT_BANNED = [
    "password", "secret", "private_key", "api_key", "token",
    "ssn", "credit_card", "cvv", "hack", "exploit", "injection",
    "drop table", "exec(", "eval(", "rm -rf", "sudo", "malware",
    "phishing", "ransomware", "bypass", "override security",
]

def compute_safety(actual: dict, banned: list | None = None) -> tuple[float, str, list[str]]:
    keywords = banned if banned else DEFAULT_BANNED
    text = json_to_text(actual).lower()
    found = [kw for kw in keywords if kw.lower() in text]
    if not found:
        return 1.0, "no unsafe content detected", []
    penalty = min(1.0, len(found) * 0.25)
    return round(max(0.0, 1.0 - penalty), 4), f"unsafe keywords detected: {found}", found


# ─────────────────────────────────────────────
# Critical Fail Rules
# ─────────────────────────────────────────────

def check_critical_fails(field_results: list[FieldResult],
                          critical_fields: list[str],
                          threshold: float = 0.30) -> list[CriticalFailReason]:
    failures: list[CriticalFailReason] = []
    fm = {r.field: r for r in field_results}

    for fld in critical_fields:
        if fld in fm and fm[fld].score < 1.0:
            failures.append(CriticalFailReason(
                rule=f"critical_field_mismatch:{fld}",
                field=fld,
                detail=f"Critical field '{fld}' score={fm[fld].score} — {fm[fld].explanation}"
            ))

    for r in field_results:
        if isinstance(r.expected, (int, float)) and isinstance(r.actual, (int, float)) \
                and r.expected != 0 and r.score == 0.0:
            dev = abs(float(r.expected) - float(r.actual)) / abs(float(r.expected))
            if dev > threshold and not any(cf.field == r.field for cf in failures):
                failures.append(CriticalFailReason(
                    rule=f"numeric_deviation_exceeded:{r.field}",
                    field=r.field,
                    detail=f"'{r.field}': {dev*100:.1f}% deviation exceeds {threshold*100:.0f}% threshold"
                ))

    return failures


# ─────────────────────────────────────────────
# Main Orchestrator  ← CHANGED: passes paragraph_threshold through
# ─────────────────────────────────────────────

def evaluate(request: EvaluationRequest, config: EvaluationConfig) -> EvaluationResult:
    expected = request.expected_output
    actual = request.actual_output
    multiple = request.multiple_outputs if request.multiple_outputs else [actual]

    correctness, field_results = compute_correctness(
        expected, actual, config.field_weights,
        paragraph_threshold=config.paragraph_threshold      # ← NEW
    )
    relevance, rel_expl = compute_relevance(expected, actual)
    completeness, comp_expl, missing = compute_completeness(expected, actual)
    consistency, cons_expl = compute_consistency(multiple)
    safety, safe_expl, unsafe = compute_safety(actual, config.banned_keywords or None)

    w = config.dimension_weights
    final = (
        w["correctness"] * correctness +
        w["relevance"]   * relevance +
        w["completeness"] * completeness +
        w["consistency"]  * consistency +
        w["safety"]       * safety
    ) * 100

    critical_fails = check_critical_fails(
        field_results, config.critical_fields, config.numeric_deviation_threshold
    )

    if critical_fails:
        verdict = "FAIL"
        reason = f"Critical rule(s) triggered: {[cf.rule for cf in critical_fails]}"
    elif final >= config.pass_threshold:
        verdict = "PASS"
        reason = f"Score {final:.2f} ≥ threshold {config.pass_threshold}"
    else:
        verdict = "FAIL"
        reason = f"Score {final:.2f} < threshold {config.pass_threshold}"

    return EvaluationResult(
        scores=DimensionScores(
            correctness=correctness, relevance=relevance, completeness=completeness,
            consistency=consistency, safety=safety, final_score=round(final, 2)
        ),
        verdict=verdict,
        verdict_reason=reason,
        field_results=field_results,
        critical_fails=critical_fails,
        missing_fields=missing,
        unsafe_keywords_found=unsafe,
        explanations={
            "correctness": f"Weighted field comparison across {len(field_results)} fields",
            "relevance": rel_expl,
            "completeness": comp_expl,
            "consistency": cons_expl,
            "safety": safe_expl,
        }
    )