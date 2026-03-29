#!/usr/bin/env bash
# ── 01_build_push.sh ──────────────────────────────────────────────────────
# Build all three Docker images and push to Artifact Registry.
#
# Images built:
#   1. both-api      — Cloud Run  (TF-IDF + DistilBERT, ~2.5 GB)
#   2. tfidf-api     — Kubernetes (TF-IDF only, ~350 MB)
#   3. distilbert-api— Kubernetes (DistilBERT only, ~2.4 GB)
#
# Usage:
#   source deployment/scripts/00_setup_env.sh
#   bash deployment/scripts/01_build_push.sh
#
# Prerequisite: models must exist locally at:
#   models/baseline/pipeline.pkl
#   models/distilbert/distilbert-mental-health/
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

# ── Enable Artifact Registry & configure Docker auth ─────────────────────
echo ""
echo "=== Step 1: Enable services & configure Docker auth ==="
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Mental health classifier images" \
    2>/dev/null || echo "(repository already exists)"

gcloud auth configure-docker "${AR_HOST}" --quiet

# ── Verify model artefacts exist ─────────────────────────────────────────
echo ""
echo "=== Step 2: Verify model artefacts ==="
if [ ! -f "models/baseline/pipeline.pkl" ]; then
    echo "ERROR: models/baseline/pipeline.pkl not found."
    echo "Run notebook 02 first to train and save the TF-IDF pipeline."
    exit 1
fi
if [ ! -d "models/distilbert/distilbert-mental-health" ]; then
    echo "ERROR: models/distilbert/distilbert-mental-health/ not found."
    echo "Run notebook 03 first to fine-tune and save DistilBERT."
    exit 1
fi
echo "✓ Model artefacts found."

# ── Build 1: Cloud Run image (both models) ────────────────────────────────
echo ""
echo "=== Step 3: Build Cloud Run image (both models) ==="
docker build \
    -f deployment/cloudrun/Dockerfile \
    -t "${IMAGE_CLOUDRUN}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "Pushing ${IMAGE_CLOUDRUN}..."
docker push "${IMAGE_CLOUDRUN}"
echo "✓ Cloud Run image pushed."

# ── Build 2: Kubernetes TF-IDF image ─────────────────────────────────────
echo ""
echo "=== Step 4: Build Kubernetes TF-IDF image ==="
docker build \
    -f deployment/kubernetes/Dockerfile.tfidf \
    -t "${IMAGE_TFIDF}" \
    .

echo "Pushing ${IMAGE_TFIDF}..."
docker push "${IMAGE_TFIDF}"
echo "✓ TF-IDF image pushed."

# ── Build 3: Kubernetes DistilBERT image ──────────────────────────────────
echo ""
echo "=== Step 5: Build Kubernetes DistilBERT image ==="
docker build \
    -f deployment/kubernetes/Dockerfile.distilbert \
    -t "${IMAGE_DISTILBERT}" \
    .

echo "Pushing ${IMAGE_DISTILBERT}..."
docker push "${IMAGE_DISTILBERT}"
echo "✓ DistilBERT image pushed."

echo ""
echo "=== All images pushed ==="
echo "  Cloud Run  : ${IMAGE_CLOUDRUN}"
echo "  TF-IDF     : ${IMAGE_TFIDF}"
echo "  DistilBERT : ${IMAGE_DISTILBERT}"
