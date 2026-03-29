#!/usr/bin/env bash
# ── 03_deploy_gke.sh ──────────────────────────────────────────────────────
# Provision a GKE cluster and deploy both model services with HPA.
#
# Architecture delivered:
#   • 1 GKE Standard cluster (e2-standard-4 nodes, autoscaling 1–5 nodes)
#   • tfidf-deployment:     2–10 pods,  256Mi–512Mi RAM each
#   • distilbert-deployment: 1–4 pods, 1.5Gi–3Gi  RAM each
#   • HPA on CPU (primary) + memory (secondary) for both
#   • GKE-native Ingress → single external IP
#
# Observable auto-scaling metrics (directly answers professor feedback):
#   kubectl get hpa -n mh-classifier -w
#   Columns: NAME | TARGETS | MINPODS | MAXPODS | REPLICAS | AGE
#   Record REPLICAS at 10s intervals during Locust burst test.
#
# Usage:
#   source deployment/scripts/00_setup_env.sh
#   bash deployment/scripts/03_deploy_gke.sh
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

# ── Enable required APIs ──────────────────────────────────────────────────
echo "=== Enabling GKE APIs ==="
gcloud services enable \
    container.googleapis.com \
    compute.googleapis.com

# ── Patch image references in manifests ──────────────────────────────────
# Replace the PROJECT_ID placeholder in YAML files with the real value
echo ""
echo "=== Patching image references in manifests ==="
sed -i "s|PROJECT_ID|${PROJECT_ID}|g" \
    deployment/kubernetes/tfidf/deployment.yaml \
    deployment/kubernetes/distilbert/deployment.yaml
echo "✓ IMAGE references updated."

# ── Create GKE cluster ────────────────────────────────────────────────────
echo ""
echo "=== Creating GKE cluster: ${GKE_CLUSTER} ==="
echo "(This takes 3–5 minutes)"

gcloud container clusters create "${GKE_CLUSTER}" \
    --zone "${ZONE}" \
    --machine-type "e2-standard-4" \
    --num-nodes 2 \
    --min-nodes 1 \
    --max-nodes 5 \
    --enable-autoscaling \
    --enable-autoprovisioning \
    --enable-autorepair \
    --enable-autoupgrade \
    --addons HttpLoadBalancing,HorizontalPodAutoscaling \
    --workload-pool "${PROJECT_ID}.svc.id.goog" \
    2>/dev/null || echo "(cluster already exists, continuing)"

# Get credentials for kubectl
gcloud container clusters get-credentials "${GKE_CLUSTER}" \
    --zone "${ZONE}"

echo "✓ Cluster ready. kubectl context set."
kubectl cluster-info

# ── Deploy Kubernetes resources ───────────────────────────────────────────
echo ""
echo "=== Applying Kubernetes manifests ==="

# Namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# TF-IDF service
kubectl apply -f deployment/kubernetes/tfidf/deployment.yaml
kubectl apply -f deployment/kubernetes/tfidf/service.yaml
kubectl apply -f deployment/kubernetes/tfidf/hpa.yaml

# DistilBERT service
kubectl apply -f deployment/kubernetes/distilbert/deployment.yaml
kubectl apply -f deployment/kubernetes/distilbert/service.yaml
kubectl apply -f deployment/kubernetes/distilbert/hpa.yaml

# Ingress (GKE HTTP(S) Load Balancer)
kubectl apply -f deployment/kubernetes/ingress.yaml

echo ""
echo "=== Waiting for deployments to become ready ==="
echo "(TF-IDF: ~20s | DistilBERT: ~60s due to model load)"

kubectl rollout status deployment/tfidf-deployment \
    -n "${GKE_NAMESPACE}" --timeout=120s

kubectl rollout status deployment/distilbert-deployment \
    -n "${GKE_NAMESPACE}" --timeout=180s

# ── Get Ingress IP ────────────────────────────────────────────────────────
echo ""
echo "=== Waiting for Ingress IP (can take 2–5 min) ==="
for i in $(seq 1 30); do
    GKE_IP=$(kubectl get ingress mh-classifier-ingress \
        -n "${GKE_NAMESPACE}" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    if [ -n "${GKE_IP}" ]; then
        break
    fi
    echo "  Waiting... (${i}/30)"
    sleep 10
done

if [ -z "${GKE_IP:-}" ]; then
    echo "WARNING: Ingress IP not yet assigned. Run:"
    echo "  kubectl get ingress -n mh-classifier"
    GKE_IP="<PENDING>"
fi

# ── Cluster state summary ─────────────────────────────────────────────────
echo ""
echo "=== Cluster state ==="
kubectl get all -n "${GKE_NAMESPACE}"
echo ""
echo "=== HPA state (watch this during load tests) ==="
kubectl get hpa -n "${GKE_NAMESPACE}"

# ── Smoke tests ───────────────────────────────────────────────────────────
if [ "${GKE_IP}" != "<PENDING>" ]; then
    echo ""
    echo "=== Smoke tests ==="
    sleep 10  # brief pause for load balancer to warm up

    echo "Health check..."
    curl -sf "http://${GKE_IP}/health" | python3 -m json.tool || true

    echo ""
    echo "TF-IDF prediction..."
    curl -sf -X POST "http://${GKE_IP}/tfidf/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "I feel completely hopeless, I cannot get out of bed"}' \
        | python3 -m json.tool || true

    echo ""
    echo "DistilBERT prediction..."
    curl -sf -X POST "http://${GKE_IP}/distilbert/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "I feel completely hopeless, I cannot get out of bed"}' \
        | python3 -m json.tool || true
fi

echo ""
echo "=== GKE deployment complete ==="
echo "  Cluster  : ${GKE_CLUSTER}"
echo "  Ingress  : http://${GKE_IP}"
echo "  TF-IDF   : http://${GKE_IP}/tfidf/predict"
echo "  DistilBERT: http://${GKE_IP}/distilbert/predict"
echo ""
echo "Watch HPA during load tests:"
echo "  kubectl get hpa -n mh-classifier -w"
echo ""
echo "Watch pod scaling:"
echo "  kubectl get pods -n mh-classifier -w"
echo ""
echo "Save for Locust:"
echo "  export GKE_URL=http://${GKE_IP}"
