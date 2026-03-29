#!/usr/bin/env bash
# ── 00_setup_env.sh ───────────────────────────────────────────────────────
# Shared environment variables for all deployment scripts.
# Source this file before running any other script:
#   source deployment/scripts/00_setup_env.sh
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── GCP project config ────────────────────────────────────────────────────
export PROJECT_ID="mh-classifier"         
export REGION="us-central1"
export ZONE="${REGION}-a"

# ── Artifact Registry ─────────────────────────────────────────────────────
export AR_REPO="mh-classifier"
export AR_HOST="${REGION}-docker.pkg.dev"
export IMAGE_BASE="${AR_HOST}/${PROJECT_ID}/${AR_REPO}"

# Image tags
export IMAGE_CLOUDRUN="${IMAGE_BASE}/both-api:latest"
export IMAGE_TFIDF="${IMAGE_BASE}/tfidf-api:latest"
export IMAGE_DISTILBERT="${IMAGE_BASE}/distilbert-api:latest"

# ── Cloud Run ─────────────────────────────────────────────────────────────
export CLOUDRUN_SERVICE="mh-classifier-cloudrun"

# ── GKE ───────────────────────────────────────────────────────────────────
export GKE_CLUSTER="mh-classifier-gke"
export GKE_NAMESPACE="mh-classifier"

# ── Cloud Functions ───────────────────────────────────────────────────────
export FUNCTIONS_RUNTIME="python311"
export FUNCTION_TFIDF="mh-predict-tfidf"
export FUNCTION_DISTILBERT="mh-predict-distilbert"
# GCS bucket for staging model artefacts (functions can't bundle 269 MB easily)
export GCS_BUCKET="${PROJECT_ID}-mh-models"

# ── Auth check ────────────────────────────────────────────────────────────
echo "Project  : ${PROJECT_ID}"
echo "Region   : ${REGION}"
echo "AR host  : ${AR_HOST}"
echo ""
echo "Checking gcloud auth..."
gcloud auth list --filter=status:ACTIVE --format="value(account)"
gcloud config set project "${PROJECT_ID}"
echo "✓ Environment ready."
