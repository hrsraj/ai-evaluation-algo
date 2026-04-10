"""
Data models for the Prompt Evaluation Engine.
Production: Pydantic v2 (pip install pydantic fastapi)
Local testing: dataclasses stdlib fallback
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvaluationRequest:
    expected_output: dict
    actual_output: dict
    multiple_outputs: list = field(default_factory=list)

@dataclass
class FullEvaluationRequest:
    request: EvaluationRequest
    config: Optional[EvaluationConfig] = None
    
@dataclass
class EvaluationConfig:
    field_weights: dict = field(default_factory=dict)
    critical_fields: list = field(default_factory=list)
    dimension_weights: dict = field(default_factory=lambda: {
        "correctness": 0.40,
        "relevance": 0.20,
        "completeness": 0.15,
        "consistency": 0.15,
        "safety": 0.10,
    })
    pass_threshold: float = 85.0
    numeric_deviation_threshold: float = 0.30
    banned_keywords: list = field(default_factory=list)
    # Strings with len >= paragraph_threshold are scored via semantic TF-IDF
    # instead of fuzzy char-matching. Lower = more fields treated as paragraphs.
    paragraph_threshold: int = 60


@dataclass
class FieldResult:
    field: str
    expected: Any
    actual: Any
    score: float
    weight: float
    explanation: str


@dataclass
class CriticalFailReason:
    rule: str
    field: str
    detail: str


@dataclass
class DimensionScores:
    correctness: float
    relevance: float
    completeness: float
    consistency: float
    safety: float
    final_score: float

    def model_dump(self):
        return {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "consistency": self.consistency,
            "safety": self.safety,
            "finalScore": self.final_score,
        }


@dataclass
class EvaluationResult:
    scores: DimensionScores
    verdict: str
    verdict_reason: str
    field_results: list
    critical_fails: list
    missing_fields: list
    unsafe_keywords_found: list
    explanations: dict