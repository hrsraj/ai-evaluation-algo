"""
semantic.py — Multi-signal semantic similarity for paragraph fields.

Scoring pipeline (all stdlib + sklearn, zero external model downloads):

  Signal 1 (45%): TF-IDF cosine, unigram+bigram — shared vocabulary
  Signal 2 (35%): Stopword-filtered Jaccard — content word overlap ratio
  Signal 3 (20%): Concept-cluster match — domain synonym groups

  Final = max(tfidf_alone, weighted_hybrid)
  max() ensures TF-IDF wins when lexical overlap is strong.

─────────────────────────────────────────────────────────────────────
HONEST SCORE RANGES (no neural embeddings):

  Situation                              Expected range
  ─────────────────────────────────────  ──────────────
  Identical text                         1.00
  Reordered / trivially paraphrased      0.75–0.95
  Same topic, partial shared vocab       0.35–0.65
  Same gist, synonym-heavy rephrasing    0.20–0.45 ← TF-IDF blind spot
  Completely different domain            < 0.15

  For true synonym-paraphrase detection (>0.70 on "declined"/"rejected")
  upgrade to sentence-transformers:

    pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    _ST = SentenceTransformer("all-MiniLM-L6-v2")   # 80MB download once
    def semantic_score(e, a):
        emb = _ST.encode([e, a])
        sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        return round(sim, 4), f"sentence-transformer cosine: {sim*100:.1f}%"
─────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Stopwords ──────────────────────────────────────────────────────
_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","need","dare","ought",
    "to","of","in","for","on","with","at","by","from","as","into",
    "through","during","before","after","above","below","between",
    "out","off","over","under","again","further","then","once",
    "and","but","or","nor","so","yet","both","either","neither",
    "not","no","nor","only","own","same","than","too","very",
    "just","because","due","this","that","these","those","it","its",
    "i","you","he","she","they","we","my","your","his","her","their",
    "our","which","who","what","when","where","why","how",
    "up","down","there","here","any","all","each","every","about",
    "also","such","while","if","else","although","though","since",
}

# ── Domain concept clusters ────────────────────────────────────────
# Rules: keep clusters SPECIFIC. Avoid generic verbs ("confirmed") that
# appear across unrelated domains. Each word must be domain-anchored.
# Extend per your AI service domain — these target banking/fintech/fraud.
_CONCEPT_CLUSTERS: list[set[str]] = [
    # Transaction outcome: failure
    {"declined", "rejected", "failed", "denied", "refused",
     "could not be processed", "not processed", "unsuccessful"},
    # Transaction outcome: success  (kept specific — no generic 'confirmed')
    {"approved", "accepted", "authorized", "successful payment",
     "transaction successful", "payment successful"},
    # Funds / balance (kept as one cluster — they're synonymous in context)
    {"insufficient funds", "insufficient balance", "balance too low",
     "not enough money", "funds unavailable", "no funds", "low balance"},
    # Account reference
    {"account", "bank account", "customer account", "wallet"},
    # Transaction entity
    {"transaction", "payment", "purchase", "transfer", "charge"},
    # Fraud / risk
    {"fraud", "fraudulent", "suspicious", "anomaly", "anomalous",
     "high risk", "unusual activity", "risk score"},
    # Flagged / alert
    {"flagged", "flagged for review", "alert", "marked", "identified"},
    # Geographic
    {"geographic", "location anomaly", "unusual location", "foreign transaction"},
    # Loan / credit (domain-specific — won't fire for satellites)
    {"loan", "credit", "mortgage", "borrowing", "advance"},
    # Verification (kept specific — paired with financial context)
    {"income verification", "credit verification", "identity verification",
     "document verification", "kyc"},
    # Spending pattern
    {"spending pattern", "typical spending", "usual spending",
     "spending behavior", "historical spending"},
]


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, remove stopwords → content word set."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _concept_score(text_a: str, text_b: str) -> float:
    """
    For each synonym cluster: check if both texts contain any member.
    score = matched_clusters / clusters_present_in_either_text.
    """
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


def semantic_score(expected_text: str, actual_text: str) -> tuple[float, str]:
    """
    Multi-signal semantic similarity for paragraph fields.
    Returns (score 0–1, explanation string).

    Transparent breakdown:
      tfidf   — vocabulary cosine (strong when shared vocab exists)
      jaccard — content-word overlap after stopword removal
      concept — domain synonym cluster matching

    Final score = max(tfidf, 0.45*tfidf + 0.35*jaccard + 0.20*concept)
    """
    if not expected_text.strip() or not actual_text.strip():
        return 0.0, "semantic: empty text"

    tfidf  = _tfidf_cosine(expected_text, actual_text)
    tok_a  = _tokenize(expected_text)
    tok_b  = _tokenize(actual_text)
    jacc   = _jaccard(tok_a, tok_b)
    concept = _concept_score(expected_text, actual_text)

    hybrid = 0.45 * tfidf + 0.35 * jacc + 0.20 * concept
    final  = round(max(tfidf, hybrid), 4)

    return final, (
        f"semantic[tfidf={tfidf:.2f} jaccard={jacc:.2f} concept={concept:.2f}]"
        f" → {final*100:.1f}%"
    )