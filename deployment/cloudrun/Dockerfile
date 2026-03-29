# ── Dockerfile.cloudrun ───────────────────────────────────────────────────
# Serves both TF-IDF and DistilBERT from a single container.
# Used for Cloud Run deployment where Google manages scaling.
# Image size: ~2.5 GB (torch + transformers + both model artefacts)
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first — leverages Docker layer cache on code-only changes
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY api/main.py         ./main.py
COPY api/model_loader.py ./model_loader.py

# Model artefacts
COPY models/baseline/   ./models/baseline/
COPY models/distilbert/ ./models/distilbert/

# Both models served by default
ENV MODEL_TYPE=both
ENV PORT=8080

# Liveness/readiness probe endpoint: GET /health
EXPOSE 8080

# --workers 1: one process = one copy of models in memory
# Cloud Run horizontal scaling (replicas) handles throughput
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
