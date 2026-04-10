# Prompt Evaluation Engine

A **deterministic, explainable, production-ready** backend system for evaluating AI-generated JSON outputs against expected ground-truth. No LLM dependency. Fast, modular, and extensible.

---

## Architecture

```
eval_engine/
├── models.py       # Pydantic v2 data models (dataclass fallback for testing)
├── evaluator.py    # Core scoring logic — all 5 dimensions
├── main.py         # FastAPI REST service (3 endpoints)
├── tests.py        # Pytest test suite (~30 test cases)
└── requirements.txt
```

---

## Scoring Dimensions

| Dimension     | Weight | Method                                   |
|---------------|--------|------------------------------------------|
| Correctness   | 40%    | Field-by-field, type-aware, weighted     |
| Relevance     | 20%    | TF-IDF + cosine similarity               |
| Completeness  | 15%    | Field presence ratio                     |
| Consistency   | 15%    | Pairwise cosine across multiple outputs  |
| Safety        | 10%    | Banned keyword detection                 |

**Final Score** = `0.4×C + 0.2×R + 0.15×Cm + 0.15×Cn + 0.1×S` × 100

---

## Numeric Penalty Bands (key fix)

| Deviation   | Score Range  | Band          |
|-------------|--------------|---------------|
| 0–5%        | 0.90–1.00    | Near-perfect  |
| 5–15%       | 0.60–0.90    | Moderate      |
| 15–30%      | 0.00–0.60    | High penalty  |
| **>30%**    | **0.00**     | **ZERO**      |

```python
# Before fix: amount=1200 vs amount=500 → high score, PASS ← BUG
# After fix:  amount=1200 vs amount=500 → score=0.0, FAIL ← CORRECT
score_numeric(1200, 500)  # → (0.0, "±58.3% deviation — exceeds 30% threshold → ZERO")
```

---

## Critical Fail Rules (Override Logic)

Critical fails force **FAIL** regardless of final score:

1. **Critical field mismatch** — any listed field with score < 1.0
2. **Numeric deviation > 30%** — automatic zero + forced FAIL

```python
config = EvaluationConfig(
    critical_fields=["fraud"],          # fraud mismatch → always FAIL
    numeric_deviation_threshold=0.30,   # >30% on any numeric → FAIL
)
```

---

## Verdict Logic

```
IF any critical_fail triggered    → FAIL  (override)
ELIF final_score >= pass_threshold → PASS
ELSE                               → FAIL
```

---

## Installation

```bash
pip install fastapi uvicorn pydantic rapidfuzz scikit-learn numpy
```

Run the API:
```bash
uvicorn main:app --reload --port 8000
```

---

## API Endpoints

### `POST /evaluate` — Full evaluation with field-level explainability

```json
{
  "request": {
    "expected_output": { "fraud": false, "amount": 1200, "category": "food" },
    "actual_output":   { "fraud": false, "amount": 1180, "category": "food" },
    "multiple_outputs": [
      { "fraud": false, "amount": 1180, "category": "food" },
      { "fraud": false, "amount": 1195, "category": "food" }
    ]
  },
  "config": {
    "field_weights": { "fraud": 0.5, "amount": 0.3, "category": 0.2 },
    "critical_fields": ["fraud"],
    "pass_threshold": 85
  }
}
```

**Response:**
```json
{
  "scores": {
    "correctness": 0.9533,
    "relevance": 0.9800,
    "completeness": 1.0,
    "consistency": 0.9900,
    "safety": 1.0,
    "final_score": 97.53
  },
  "verdict": "PASS",
  "verdict_reason": "Score 97.53 ≥ threshold 85.0",
  "field_results": [
    { "field": "fraud",    "expected": false, "actual": false, "score": 1.0,  "explanation": "exact boolean match" },
    { "field": "amount",   "expected": 1200,  "actual": 1180,  "score": 0.9833, "explanation": "±1.7% deviation (within 5%)" },
    { "field": "category", "expected": "food","actual": "food", "score": 1.0, "explanation": "exact string match" }
  ],
  "critical_fails": [],
  "missing_fields": [],
  "unsafe_keywords_found": [],
  "explanations": {
    "correctness": "Weighted field comparison across 3 fields",
    "relevance": "TF-IDF cosine similarity: 98.0%",
    "completeness": "3/3 expected fields present",
    "consistency": "avg pairwise cosine across 2 outputs: 99.0%",
    "safety": "no unsafe content detected"
  }
}
```

### `POST /evaluate/quick` — Scores + verdict only (lightweight)

### `POST /evaluate/batch` — Evaluate multiple pairs in one request

---

## Key Design Scenarios

### Scenario A: Known Bug (Large Numeric Deviation)
```python
# expected: {"amount": 1200}, actual: {"amount": 500}
# deviation = 58.3% > 30% → score_numeric returns 0.0
# correctness = 0.0 → final_score ~47 → FAIL ✓
```

### Scenario B: Fraud Flag Override
```python
# expected: {"fraud": True}, actual: {"fraud": False}
# config.critical_fields = ["fraud"]
# → CriticalFailReason triggered → verdict = FAIL ✓ (even if score is high)
```

### Scenario C: Field Weights (Finance Example)
```python
config = EvaluationConfig(
    field_weights={"fraud": 0.5, "amount": 0.3, "category": 0.2}
)
# fraud mismatch hurts 5× more than category mismatch
```

### Scenario D: Safety — Sensitive Data Leak
```python
# actual: {"note": "api_key=xyz123"}
# → safety_score = 0.75, unsafe_keywords_found = ["api_key"]
```

---

## Extending the Engine

**Add a new scoring dimension:**
```python
# In evaluator.py — add compute_* function, wire into evaluate()
# Update dimension_weights in EvaluationConfig
```

**Add new critical fail rules:**
```python
# In check_critical_fails() — add rule block with CriticalFailReason
```

**Add new banned keywords:**
```python
config = EvaluationConfig(banned_keywords=["internal_only", "do_not_share"])
```

**Integrate rapidfuzz (production):**
```bash
pip install rapidfuzz
# evaluator.py auto-detects and uses it via the try/except import block
```

---

## Running Tests

```bash
pytest tests.py -v --tb=short
```

All 30 test cases cover: numeric bands, boolean matching, fuzzy strings, missing fields, field weights, consistency across outputs, safety detection, critical fail overrides, and full integration scenarios.

---

## Design Principles

- **Deterministic** — same input always produces same output
- **No LLM dependency** — pure algorithmic evaluation
- **Explainable** — every field has a score + human-readable explanation
- **Strict on numbers** — 30% threshold is a hard ceiling, not a guideline
- **Override rules** — critical fails bypass scoring entirely
- **Extensible** — add dimensions, rules, or weights without refactoring
