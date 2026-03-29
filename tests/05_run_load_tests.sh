#!/usr/bin/env bash
# ── 05_run_load_tests.sh ──────────────────────────────────────────────────
# Runs all Locust load test scenarios for the GKE vs Cloud Run comparison.
#
# Configurations: 2 platforms x 2 models x 2 traffic patterns = 8 runs
# Plus: 2 cold-start probes (one per model on Cloud Run after idle)
# Total: 10 test runs
#
# Metrics captured per run (results/<name>_stats.csv):
#   p50 / p95 / p99 response time (ms)
#   requests/sec (throughput)
#   failure_count / total_requests (error rate %)
#
# GKE auto-scaling metric (answers professor feedback directly):
#   results/gke_hpa_burst.log -- REPLICAS at 5s intervals during burst
#   Scale-up latency computed by 06_analyze_results.py from this log.
#
# Usage:
#   source deployment/scripts/00_setup_env.sh
#   export GKE_URL=http://<your-gke-ingress-ip>
#   export CLOUDRUN_URL=https://<your-cloudrun-url>
#   bash tests/05_run_load_tests.sh
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail
source "$(dirname "$0")/../deployment/scripts/00_setup_env.sh"

: "${GKE_URL:?GKE_URL not set. Run 03_deploy_gke.sh first.}"
: "${CLOUDRUN_URL:?CLOUDRUN_URL not set. Run 02_deploy_cloudrun.sh first.}"

pip install locust --quiet

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"

# ── Helper ────────────────────────────────────────────────────────────────
run_locust() {
    local class=$1
    local name=$2
    local users=$3
    local rate=$4
    local duration=$5

    echo ""
    echo "── ${name} | users=${users} rate=${rate}/s duration=${duration} ──"

    locust \
        -f tests/locustfile.py "${class}" \
        --headless \
        -u "${users}" -r "${rate}" \
        --run-time "${duration}" \
        --csv "${RESULTS_DIR}/${name}" \
        --html "${RESULTS_DIR}/${name}.html" \
        --logfile "${RESULTS_DIR}/${name}.log" \
        2>&1 | tail -5

    echo "  -> ${RESULTS_DIR}/${name}_stats.csv"
}

# ── HPA watcher (runs in background during GKE burst tests) ───────────────
watch_hpa() {
    local log_file=$1
    echo "timestamp,tfidf_replicas,distilbert_replicas" > "${log_file}"
    while true; do
        local ts
        ts=$(date +%s)
        local tr dr
        tr=$(kubectl get hpa tfidf-hpa -n mh-classifier \
            -o jsonpath='{.status.currentReplicas}' 2>/dev/null || echo "?")
        dr=$(kubectl get hpa distilbert-hpa -n mh-classifier \
            -o jsonpath='{.status.currentReplicas}' 2>/dev/null || echo "?")
        echo "${ts},${tr},${dr}" >> "${log_file}"
        sleep 5
    done
}

# ─────────────────────────────────────────────────────────────────────────
echo "======================================================="
echo " PHASE A: STEADY TRAFFIC  (1 req/sec, 60s)"
echo " Purpose: baseline warm-path latency, no scaling needed"
echo "======================================================="

run_locust GkeTfIdf          "gke_tfidf_steady"          1  1  "60s"
run_locust GkeDistilBert     "gke_distilbert_steady"      1  1  "60s"
run_locust CloudRunTfIdf     "cloudrun_tfidf_steady"      1  1  "60s"
run_locust CloudRunDistilBert "cloudrun_distilbert_steady" 1  1  "60s"

# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo " PHASE B: BURST TRAFFIC  (0->100 users, 10s ramp, 90s hold)"
echo " Purpose: auto-scaling, tail latency, error rate under load"
echo " GKE: HPA replica count logged every 5s to results/gke_hpa_burst.log"
echo "======================================================="

HPA_LOG="${RESULTS_DIR}/gke_hpa_burst.log"
watch_hpa "${HPA_LOG}" &
HPA_PID=$!
echo "HPA watcher started (PID=${HPA_PID})"

run_locust GkeTfIdf           "gke_tfidf_burst"           100 10 "100s"
run_locust GkeDistilBert      "gke_distilbert_burst"       100 10 "100s"
run_locust CloudRunTfIdf      "cloudrun_tfidf_burst"       100 10 "100s"
run_locust CloudRunDistilBert "cloudrun_distilbert_burst"  100 10 "100s"

kill "${HPA_PID}" 2>/dev/null || true
echo "HPA log saved: ${HPA_LOG}"

# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo " PHASE C: COLD START PROBES (Cloud Run only)"
echo " Purpose: measure cold-start overhead per model"
echo " GKE is excluded -- pods are always warm by design"
echo " Waiting 15 minutes for Cloud Run to go idle..."
echo " (Press Ctrl+C to skip)"
echo "======================================================="

sleep 900  # Cloud Run recycles instances after ~10 min idle

echo "Probing Cloud Run TF-IDF cold start..."
run_locust ColdStartProbe "cloudrun_tfidf_coldstart"       1  1  "30s"

echo "Waiting 5 minutes for Cloud Run to go idle again..."
sleep 300

# Switch probe to DistilBERT -- edit ColdStartProbe.ENDPOINT in locustfile.py
# or pass it via env var. Here we use sed for a clean one-shot change.
sed -i 's|ENDPOINT = "/predict?model=tfidf"|ENDPOINT = "/predict?model=distilbert"|' \
    tests/locustfile.py

echo "Probing Cloud Run DistilBERT cold start..."
run_locust ColdStartProbe "cloudrun_distilbert_coldstart"  1  1  "60s"

# Restore original endpoint
sed -i 's|ENDPOINT = "/predict?model=distilbert"|ENDPOINT = "/predict?model=tfidf"|' \
    tests/locustfile.py

# ─────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo " QUICK SUMMARY"
echo "======================================================="
python3 - <<'EOF'
import csv, glob, os
files = sorted(glob.glob("results/*_stats.csv"))
print(f"\n{'Config':<42} {'p50':>5} {'p95':>5} {'p99':>5} {'RPS':>6} {'Err%':>6}")
print("-" * 70)
for f in files:
    name = os.path.basename(f).replace("_stats.csv", "")
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row.get("Name") == "Aggregated":
                n = int(row.get("Request Count", 0))
                e = int(row.get("Failure Count", 0))
                err = f"{e/n*100:.1f}" if n else "N/A"
                print(
                    f"{name:<42} "
                    f"{float(row.get('50%',0)):>5.0f} "
                    f"{float(row.get('95%',0)):>5.0f} "
                    f"{float(row.get('99%',0)):>5.0f} "
                    f"{float(row.get('Requests/s',0)):>6.1f} "
                    f"{err:>6}"
                )
EOF

echo ""
echo "Full results in: ${RESULTS_DIR}/"
echo "HTML reports:    ${RESULTS_DIR}/*.html"
echo "GKE HPA log:     ${HPA_LOG}"
echo ""
echo "Next: python3 tests/06_analyze_results.py"
