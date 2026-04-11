"""
models.py — Pydantic v2 data models for the Prompt Evaluation Engine.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    """Input payload for a single evaluation run."""

    expected_output: dict[str, Any] = Field(
        ...,
        description="Ground-truth JSON the AI service should have produced.",
    )
    actual_output: dict[str, Any] = Field(
        ...,
        description="Actual JSON produced by the AI service under evaluation.",
    )
    multiple_outputs: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional list of repeated outputs for the same prompt. "
            "Used to score consistency across runs. "
            "If omitted, consistency defaults to 1.0."
        ),
    )


class EvaluationConfig(BaseModel):
    """
    Full configuration for one evaluation run.
    All fields have safe production defaults.
    """

    # ── Correctness ──────────────────────────────────────────────────
    field_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-field importance weights for correctness scoring. "
            "Keys are dot-separated flattened field paths (e.g. 'meta.score'). "
            "Unspecified fields default to weight 1.0. "
            "Example: {'fraud': 0.5, 'amount': 0.3, 'category': 0.2}"
        ),
    )

    paragraph_threshold: int = Field(
        default=60,
        ge=10,
        description=(
            "Character length above which a string field is treated as a "
            "paragraph and scored with sentence-transformer embeddings instead "
            "of fuzzy token matching. Default 60 chars ≈ one short sentence."
        ),
    )

    # ── Critical fail overrides ──────────────────────────────────────
    critical_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Field names that must match exactly (score = 1.0). "
            "Any mismatch on these fields forces verdict = FAIL regardless "
            "of the overall score. Typical use: ['fraud', 'risk_level']."
        ),
    )
    numeric_deviation_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Fractional deviation above which a numeric field score is forced "
            "to 0.0 and triggers a critical fail. Default 0.30 = 30%."
        ),
    )

    # ── Dimension weights ────────────────────────────────────────────
    dimension_weights: dict[str, float] = Field(
        default={
            "correctness":  0.40,
            "relevance":    0.20,
            "completeness": 0.15,
            "consistency":  0.15,
            "safety":       0.10,
        },
        description="Weights for the five scoring dimensions. Must sum to 1.0.",
    )

    # ── Verdict ──────────────────────────────────────────────────────
    pass_threshold: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Minimum final score (0–100) required for PASS verdict.",
    )

    # ── Safety ───────────────────────────────────────────────────────
    banned_keywords: Optional[list[str]] = Field(
        default=None,
        description=(
            "Custom banned keyword list for safety scoring. "
            "If None, the engine's built-in default list is used."
        ),
    )

    @model_validator(mode="after")
    def _validate_dimension_weights(self) -> "EvaluationConfig":
        total = sum(self.dimension_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"dimension_weights must sum to 1.0, got {total:.4f}"
            )
        required = {"correctness", "relevance", "completeness", "consistency", "safety"}
        missing = required - self.dimension_weights.keys()
        if missing:
            raise ValueError(f"dimension_weights missing keys: {missing}")
        return self


class FullEvaluationRequest(BaseModel):
    """Top-level request envelope sent to the /evaluate endpoint."""

    request: EvaluationRequest
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)


class BatchEvaluationRequest(BaseModel):
    """Request envelope for /evaluate/batch."""

    items: list[FullEvaluationRequest] = Field(
        ..., min_length=1, max_length=100
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result Models
# ─────────────────────────────────────────────────────────────────────────────

class FieldResult(BaseModel):
    """Score and explanation for one flattened JSON field."""

    field: str = Field(description="Dot-separated field path, e.g. 'meta.score'.")
    expected: Any = Field(description="Ground-truth value.")
    actual: Any = Field(description="Actual value (None if field was missing).")
    score: float = Field(ge=0.0, le=1.0, description="Field-level score 0–1.")
    weight: float = Field(ge=0.0, description="Field weight used in correctness rollup.")
    explanation: str = Field(description="Human-readable reason for this score.")


class CriticalFailReason(BaseModel):
    """A triggered critical fail rule that forces verdict = FAIL."""

    rule: str = Field(description="Rule identifier, e.g. 'critical_field_mismatch:fraud'.")
    field: str = Field(description="Field that triggered the rule.")
    detail: str = Field(description="Human-readable explanation.")


class DimensionScores(BaseModel):
    """Scores for each evaluation dimension plus the weighted final score."""

    correctness:  float = Field(ge=0.0, le=1.0)
    relevance:    float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    consistency:  float = Field(ge=0.0, le=1.0)
    safety:       float = Field(ge=0.0, le=1.0)
    final_score:  float = Field(ge=0.0, le=100.0)


class EvaluationResult(BaseModel):
    """Full evaluation result returned by the engine."""

    scores: DimensionScores
    verdict: str = Field(
        description="'PASS' or 'FAIL'.",
        pattern="^(PASS|FAIL)$",
    )
    verdict_reason: str = Field(
        description="One-line explanation of why the verdict was reached."
    )
    field_results: list[FieldResult] = Field(
        description="Per-field breakdown driving the correctness score."
    )
    critical_fails: list[CriticalFailReason] = Field(
        description="Critical fail rules that were triggered (empty on PASS)."
    )
    missing_fields: list[str] = Field(
        description="Expected fields absent from the actual output."
    )
    unsafe_keywords_found: list[str] = Field(
        description="Banned keywords detected in the actual output."
    )
    explanations: dict[str, str] = Field(
        description="One-line explanation per scoring dimension."
    )


class BatchResultItem(BaseModel):
    """One result inside a /evaluate/batch response."""

    index: int
    verdict: str
    scores: DimensionScores
    verdict_reason: str
    critical_fails: list[CriticalFailReason]
    error: Optional[str] = None