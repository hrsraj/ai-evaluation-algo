# ═══════════════════════════════════════════════════════════════
# Prompt Evaluation Engine — Dockerfile
# Multi-stage build: slim final image, no build tools in prod
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: dependency builder ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some wheels (numpy, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all deps into a prefix we can copy cleanly
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: sentence-transformer model pre-download ───────────
# Bake the model into the image so the first request is fast.
# Remove this stage if you want a smaller image and are OK with
# the model downloading on first startup (~80 MB, ~10s).
FROM builder AS model-fetcher

RUN pip install --no-cache-dir sentence-transformers==3.3.1 \
 && python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/model-cache')"


# ── Stage 3: production runtime ────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded model from model-fetcher
COPY --from=model-fetcher /model-cache /home/appuser/.cache/torch/sentence_transformers

# Copy application source
COPY main.py evaluator.py models.py semantic.py ./

# Own everything as non-root user
RUN chown -R appuser:appgroup /app /home/appuser

USER appuser

# Expose API port
EXPOSE 8000

# Health check — hits /health every 30s, marks unhealthy after 3 failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
  || exit 1

# Default: single worker. Override via environment for multi-core.
# WORKERS=4 docker compose up  → or set in docker-compose.yml
ENV WORKERS=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=info \
    # Tell sentence-transformers where the baked-in model lives
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/torch/sentence_transformers

CMD ["sh", "-c", \
  "uvicorn main:app \
     --host $HOST \
     --port $PORT \
     --workers $WORKERS \
     --log-level $LOG_LEVEL \
     --access-log"]