"""
main.py — FastAPI service for the Prompt Evaluation Engine.

Endpoints:
  GET  /health            — liveness + active scorer name
  POST /evaluate          — full evaluation with field-level explainability
  POST /evaluate/quick    — scores + verdict only (lower payload)
  POST /evaluate/batch    — evaluate up to 100 pairs in one request
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from evaluator import evaluate
from models import (
    BatchEvaluationRequest,
    BatchResultItem,
    EvaluationConfig,
    EvaluationResult,
    FullEvaluationRequest,
)
from semantic import active_scorer_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: warm up the semantic model once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer = active_scorer_name()
    logger.info("Semantic scorer active: %s", scorer)
    yield
    logger.info("Shutting down Prompt Evaluation Engine.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Prompt Evaluation Engine",
    description=(
        "Deterministic, explainable evaluation of AI-generated JSON outputs. "
        "Scores across Correctness, Relevance, Completeness, Consistency, and Safety. "
        "No LLM dependency. Sentence-transformer embeddings for paragraph fields."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request timing middleware
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="Liveness check",
)
def health():
    """Returns service status and the active paragraph scorer name."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "semantic_scorer": active_scorer_name(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /evaluate — full result
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/evaluate",
    response_model=EvaluationResult,
    tags=["Evaluation"],
    summary="Full evaluation with field-level explainability",
)
def evaluate_endpoint(payload: FullEvaluationRequest) -> EvaluationResult:
    """
    Evaluate one AI output against the expected output.

    Returns the complete result including per-field scores, critical fail
    reasons, missing fields, unsafe keywords, and dimension explanations.

    **Scoring dimensions (configurable weights):**
    - **Correctness (40%)** — field-by-field, type-aware, weighted
    - **Relevance (20%)** — document TF-IDF cosine
    - **Completeness (15%)** — fraction of expected fields present
    - **Consistency (15%)** — pairwise similarity across multiple_outputs
    - **Safety (10%)** — banned keyword detection

    **Verdict logic:**
    - `FAIL` if any critical rule fires (fraud mismatch, numeric deviation > 30%)
    - `PASS` if final_score ≥ pass_threshold (default 85)
    - `FAIL` otherwise
    """
    try:
        return evaluate(payload.request, payload.config)
    except Exception as exc:
        logger.exception("Evaluation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /evaluate/quick — lightweight response
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/evaluate/quick",
    tags=["Evaluation"],
    summary="Quick evaluation — scores and verdict only",
)
def evaluate_quick(payload: FullEvaluationRequest) -> dict:
    """
    Same evaluation pipeline as `/evaluate` but returns only the top-level
    scores, verdict, and verdict reason — no field-level detail.
    Suitable for high-throughput pipelines where explainability is optional.
    """
    try:
        result = evaluate(payload.request, payload.config)
        return {
            "scores": result.scores.model_dump(),
            "verdict": result.verdict,
            "verdict_reason": result.verdict_reason,
        }
    except Exception as exc:
        logger.exception("Quick evaluation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /evaluate/batch — up to 100 pairs
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/evaluate/batch",
    response_model=list[BatchResultItem],
    tags=["Evaluation"],
    summary="Batch evaluate up to 100 request-config pairs",
)
def evaluate_batch(payload: BatchEvaluationRequest) -> list[BatchResultItem]:
    """
    Evaluate a list of (request, config) pairs in a single HTTP call.
    Each item is evaluated independently. Errors in individual items are
    returned as `BatchResultItem.error` rather than failing the whole batch.
    """
    results: list[BatchResultItem] = []
    for i, item in enumerate(payload.items):
        try:
            result = evaluate(item.request, item.config)
            results.append(
                BatchResultItem(
                    index=i,
                    verdict=result.verdict,
                    scores=result.scores,
                    verdict_reason=result.verdict_reason,
                    critical_fails=result.critical_fails,
                )
            )
        except Exception as exc:
            logger.error("Batch item %d failed: %s", i, exc)
            results.append(
                BatchResultItem(
                    index=i,
                    verdict="FAIL",
                    scores=None,  # type: ignore[arg-type]
                    verdict_reason="evaluation error",
                    critical_fails=[],
                    error=str(exc),
                )
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # set True only in local dev
        workers=1,             # increase for multi-core; sentence-transformer is thread-safe
        log_level="info",
    )