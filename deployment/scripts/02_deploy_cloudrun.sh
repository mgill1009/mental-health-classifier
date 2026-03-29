#!/usr/bin/env bash
# ── 02_deploy_cloudrun.sh ─────────────────────────────────────────────────
# Deploy the both-models image to Cloud Run.
#
# Architectural properties tested here:
#   • Model loaded ONCE at container startup (not per-request)
#   • Google manages horizontal scaling: min-instances controls cold starts
#   • concurrency=1 ensures requests don't share a Python process
#     (avoids GIL contention; Cloud Run scales out instead)
#   • Memory: 4Gi needed for torch + DistilBERT + headroom
#
# Produces:
#   CLOUDRUN_URL   — the HTTPS endpoint for load testing
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

echo ""
echo "=== Deploying to Cloud Run ==="
gcloud services enable run.googleapis.com

gcloud run deploy "${CLOUDRUN_SERVICE}" \
    --image "${IMAGE_CLOUDRUN}" \
    --platform managed \
    --region "${REGION}" \
    --memory 4Gi \
    --cpu 2 \
    --timeout 60 \
    --concurrency 1 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars MODEL_TYPE=both \
    --allow-unauthenticated \
    --port 8080

# ── Capture endpoint URL ──────────────────────────────────────────────────
CLOUDRUN_URL=$(gcloud run services describe "${CLOUDRUN_SERVICE}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo ""
echo "✓ Cloud Run deployed."
echo "  URL: ${CLOUDRUN_URL}"
echo ""

# ── Smoke test ────────────────────────────────────────────────────────────
echo "=== Smoke test ==="

echo "Health check..."
curl -sf "${CLOUDRUN_URL}/health" | python3 -m json.tool

echo ""
echo "TF-IDF prediction..."
curl -sf -X POST "${CLOUDRUN_URL}/predict?model=tfidf" \
    -H "Content-Type: application/json" \
    -d '{"text": "I feel completely hopeless, I cannot get out of bed"}' \
    | python3 -m json.tool

echo ""
echo "DistilBERT prediction..."
curl -sf -X POST "${CLOUDRUN_URL}/predict?model=distilbert" \
    -H "Content-Type: application/json" \
    -d '{"text": "I feel completely hopeless, I cannot get out of bed"}' \
    | python3 -m json.tool

echo ""
echo "=== Cloud Run deployment complete ==="
echo "Save this URL for Locust load tests:"
echo "  export CLOUDRUN_URL=${CLOUDRUN_URL}"
