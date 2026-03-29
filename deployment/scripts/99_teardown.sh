#!/usr/bin/env bash
# ── 99_teardown.sh ────────────────────────────────────────────────────────
# Deletes all GCP resources created by this project.
# Run this after load testing is complete to avoid ongoing charges.
#
# Resources deleted:
#   - Cloud Run service
#   - GKE cluster (most expensive — ~$0.27/hr for 2 nodes)
#   - Cloud Functions (both)
#   - Artifact Registry images
#
# Resources NOT deleted (kept for re-use):
#   - Artifact Registry repository (empty, costs ~$0)
#   - GCS bucket (if created for Option B model staging)
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

echo ""
echo "⚠️  This will DELETE all GCP resources for project: ${PROJECT_ID}"
echo "    Cloud Run service, GKE cluster, Cloud Functions, AR images."
echo ""
read -p "Type 'yes' to confirm: " CONFIRM
if [ "${CONFIRM}" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=== Deleting Cloud Run service ==="
gcloud run services delete "${CLOUDRUN_SERVICE}" \
    --region "${REGION}" --quiet 2>/dev/null || echo "(already deleted)"

echo ""
echo "=== Deleting GKE cluster ==="
gcloud container clusters delete "${GKE_CLUSTER}" \
    --zone "${ZONE}" --quiet 2>/dev/null || echo "(already deleted)"

echo ""
echo "=== Deleting Cloud Functions ==="
gcloud functions delete "${FUNCTION_TFIDF}" \
    --gen2 --region "${REGION}" --quiet 2>/dev/null || echo "(already deleted)"

gcloud functions delete "${FUNCTION_DISTILBERT}" \
    --gen2 --region "${REGION}" --quiet 2>/dev/null || echo "(already deleted)"

echo ""
echo "=== Deleting Artifact Registry images ==="
gcloud artifacts docker images delete "${IMAGE_CLOUDRUN}"    --quiet 2>/dev/null || true
gcloud artifacts docker images delete "${IMAGE_TFIDF}"       --quiet 2>/dev/null || true
gcloud artifacts docker images delete "${IMAGE_DISTILBERT}"  --quiet 2>/dev/null || true

echo ""
echo "✓ Teardown complete. All billable resources deleted."
echo "  Check GCP console to confirm no unexpected resources remain."
