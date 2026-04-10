"""
Prompt Evaluation Engine — FastAPI Service
Production-ready, deterministic AI output scoring.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from models import FullEvaluationRequest, EvaluationResult, EvaluationConfig
from evaluator import evaluate

app = FastAPI(
    title="Prompt Evaluation Engine",
    description=(
        "A deterministic, explainable evaluation system for AI-generated JSON outputs. "
        "Scores across Correctness, Relevance, Completeness, Consistency, and Safety."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "service": "Prompt Evaluation Engine", "version": "1.0.0"}


# ─────────────────────────────────────────────
# Core Evaluation Endpoint
# ─────────────────────────────────────────────

@app.post(
    "/evaluate",
    response_model=EvaluationResult,
    tags=["Evaluation"],
    summary="Evaluate an AI output against expected output",
)
def evaluate_output(payload: FullEvaluationRequest) -> EvaluationResult:
    """
    Evaluate an AI-generated JSON output against an expected JSON output.

    **Scoring Dimensions:**
    - **Correctness (40%)** — Field-by-field comparison with type-aware scoring.
    - **Relevance (20%)** — TF-IDF cosine similarity between expected and actual.
    - **Completeness (15%)** — Fraction of expected fields present in actual.
    - **Consistency (15%)** — Pairwise similarity across multiple outputs (if provided).
    - **Safety (10%)** — Banned keyword detection in actual output.

    **Verdict Logic:**
    - `FAIL` if any critical rule is triggered (e.g., fraud field mismatch, numeric deviation > 30%).
    - `PASS` if `finalScore >= pass_threshold` (default 85).
    - `FAIL` otherwise.
    """
    try:
        config = payload.config or EvaluationConfig()
        result = evaluate(payload.request, config)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


# ─────────────────────────────────────────────
# Quick Evaluate (defaults only, minimal payload)
# ─────────────────────────────────────────────

@app.post(
    "/evaluate/quick",
    response_model=dict,
    tags=["Evaluation"],
    summary="Quick evaluation with default config — returns scores and verdict only",
)
def quick_evaluate(payload: FullEvaluationRequest):
    """
    Lightweight endpoint returning only scores and verdict (no field-level detail).
    """
    try:
        config = payload.config or EvaluationConfig()
        result = evaluate(payload.request, config)
        return {
            "scores": result.scores.model_dump(),
            "verdict": result.verdict,
            "verdict_reason": result.verdict_reason,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


# ─────────────────────────────────────────────
# Batch Evaluation
# ─────────────────────────────────────────────

@app.post(
    "/evaluate/batch",
    response_model=list[dict],
    tags=["Evaluation"],
    summary="Evaluate multiple request-config pairs in one call",
)
def batch_evaluate(payloads: list[FullEvaluationRequest]):
    """
    Evaluate a list of evaluation requests. Each item can have its own config.
    Returns a list of score summaries with index and verdict.
    """
    results = []
    for i, payload in enumerate(payloads):
        try:
            config = payload.config or EvaluationConfig()
            result = evaluate(payload.request, config)
            results.append({
                "index": i,
                "scores": result.scores.model_dump(),
                "verdict": result.verdict,
                "verdict_reason": result.verdict_reason,
                "critical_fails": [cf.model_dump() for cf in result.critical_fails],
            })
        except Exception as e:
            results.append({"index": i, "error": str(e), "verdict": "ERROR"})
    return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
