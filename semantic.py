"""
semantic.py — Sentence-level semantic similarity for paragraph fields.

Primary scorer:  sentence-transformers (all-MiniLM-L6-v2)
                 384-dim dense embeddings, cosine similarity.
                 Handles synonyms, paraphrases, and cross-vocabulary
                 meaning equivalence natively.

Fallback scorer: TF-IDF bigram cosine + stopword-filtered Jaccard
                 + domain concept-cluster matching.
                 Used automatically when sentence-transformers is not
                 installed or the model fails to load.

Install for primary scorer:
    pip install sentence-transformers torch

The fallback activates silently — no code changes needed.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-Transformers Primary Scorer
# ─────────────────────────────────────────────────────────────────────────────

_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_st_scorer: Callable[[str, str], tuple[float, str]] | None = None
_st_load_attempted = False


def _load_sentence_transformer() -> bool:
    """
    Attempt to load the sentence-transformer model once.
    Returns True on success, False if the library is unavailable.
    Caches the loaded model in module-level _st_scorer.
    """
    global _st_scorer, _st_load_attempted
    if _st_load_attempted:
        return _st_scorer is not None
    _st_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        model = SentenceTransformer(_ST_MODEL_NAME)
        logger.info("sentence-transformers loaded: %s", _ST_MODEL_NAME)

        def _scorer(text_a: str, text_b: str) -> tuple[float, str]:
            embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
            sim = float(_cos([embeddings[0]], [embeddings[1]])[0][0])
            return round(sim, 4), f"sentence-transformer({_ST_MODEL_NAME}) cosine: {sim*100:.1f}%"

        _st_scorer = _scorer
        return True

    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Falling back to TF-IDF + Jaccard + concept-cluster scorer. "
            "Run: pip install sentence-transformers torch"
        )
        return False
    except Exception as exc:
        logger.error("Failed to load sentence-transformer model: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: TF-IDF + Jaccard + Domain Concept Clusters
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "only", "own", "same", "than", "too", "very", "just",
    "because", "due", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "they", "we", "my", "your", "his", "her",
    "their", "our", "which", "who", "what", "when", "where", "why", "how",
    "up", "down", "there", "here", "any", "all", "each", "every", "about",
    "also", "such", "while", "if", "else", "although", "though", "since",
})

# Domain synonym clusters for fintech/banking/fraud AI services.
# Each set = group of words/phrases treated as semantically equivalent.
# Extend this list for your specific domain (healthcare, legal, e-commerce…).
_CONCEPT_CLUSTERS: list[frozenset[str]] = [
    frozenset({"declined", "rejected", "failed", "denied", "refused",
               "could not be processed", "not processed", "unsuccessful"}),
    frozenset({"approved", "accepted", "authorized", "successful payment",
               "transaction successful", "payment successful"}),
    frozenset({"insufficient funds", "insufficient balance", "balance too low",
               "not enough money", "funds unavailable", "no funds", "low balance"}),
    frozenset({"account", "bank account", "customer account", "wallet"}),
    frozenset({"transaction", "payment", "purchase", "transfer", "charge"}),
    frozenset({"fraud", "fraudulent", "suspicious", "anomaly", "anomalous",
               "high risk", "unusual activity", "risk score"}),
    frozenset({"flagged", "flagged for review", "marked", "identified as"}),
    frozenset({"geographic anomaly", "location anomaly", "unusual location",
               "foreign transaction", "out-of-region"}),
    frozenset({"loan", "credit", "mortgage", "borrowing", "advance"}),
    frozenset({"income verification", "credit verification", "identity verification",
               "document verification", "kyc", "know your customer"}),
    frozenset({"spending pattern", "typical spending", "usual spending",
               "spending behavior", "historical spending", "spending history"}),
]


def _tokenize(text: str) -> frozenset[str]:
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _concept_overlap(text_a: str, text_b: str) -> float:
    ta, tb = text_a.lower(), text_b.lower()
    matched = total = 0
    for cluster in _CONCEPT_CLUSTERS:
        in_a = any(w in ta for w in cluster)
        in_b = any(w in tb for w in cluster)
        if in_a or in_b:
            total += 1
            if in_a and in_b:
                matched += 1
    return matched / total if total > 0 else 0.0


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        mat = vec.fit_transform([text_a, text_b])
        return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0


def _fallback_scorer(text_a: str, text_b: str) -> tuple[float, str]:
    """
    TF-IDF cosine + Jaccard + concept-cluster hybrid.
    Score bands:
      Identical / trivially reordered  → 0.75–1.00
      Same topic, shared vocab         → 0.35–0.70
      Same gist, synonym-heavy         → 0.20–0.45  ← TF-IDF blind spot
      Different domain                 → < 0.15
    """
    tfidf   = _tfidf_cosine(text_a, text_b)
    tok_a   = _tokenize(text_a)
    tok_b   = _tokenize(text_b)
    jacc    = _jaccard(tok_a, tok_b)
    concept = _concept_overlap(text_a, text_b)

    hybrid = 0.45 * tfidf + 0.35 * jacc + 0.20 * concept
    final  = round(max(tfidf, hybrid), 4)
    return final, (
        f"fallback[tfidf={tfidf:.2f} jaccard={jacc:.2f} concept={concept:.2f}]"
        f" → {final*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def semantic_score(expected_text: str, actual_text: str) -> tuple[float, str]:
    """
    Compute semantic similarity between two text passages.
    Returns (score: float 0–1, explanation: str).

    Uses sentence-transformers if available (install: pip install sentence-transformers torch).
    Automatically falls back to TF-IDF + Jaccard + concept-cluster hybrid otherwise.

    The active scorer is chosen once at first call and cached for the process lifetime.
    """
    if not expected_text.strip() or not actual_text.strip():
        return 0.0, "semantic: one or both texts are empty"

    if expected_text.strip() == actual_text.strip():
        return 1.0, "semantic: exact match"

    # Try primary (sentence-transformers)
    if _load_sentence_transformer() and _st_scorer is not None:
        try:
            return _st_scorer(expected_text, actual_text)
        except Exception as exc:
            logger.warning("sentence-transformer inference failed (%s), using fallback", exc)

    # Fallback
    return _fallback_scorer(expected_text, actual_text)


def active_scorer_name() -> str:
    """Return which scorer is active. Useful for health-check and logging."""
    _load_sentence_transformer()
    return _ST_MODEL_NAME if _st_scorer is not None else "tfidf-jaccard-concept-fallback"