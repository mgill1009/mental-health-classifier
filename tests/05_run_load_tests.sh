#!/usr/bin/env bash
# ── 05_run_load_tests.sh ──────────────────────────────────────────────────
#
# Complete load test suite for GKE vs Cloud Run comparison.
#
# Phases:
#   A  — Steady traffic     (1 req/sec, 60s)           4 configs
#   B1 — Burst traffic      (100 users, 10/sec, 100s)  4 configs + HPA log
#   B2 — Medium load        (20 users, 2/sec, 120s)    2 configs (DistilBERT)
#   C  — Cold-start probes  (after 15 min idle)        2 configs
#
# Usage:
#   source deployment/scripts/00_setup_env.sh
#   export GKE_URL="http://<gke-ingress-ip>"
#   export CLOUDRUN_URL="https://<cloudrun-url>"
#   export TFIDF_SERVICE_URL="https://<mh-classifier-tfidf-url>"
#   export DISTILBERT_SERVICE_URL="https://<mh-classifier-distilbert-url>"
#   bash tests/05_run_load_tests.sh
#
# ─────────────────────────────────────────────────────────────────────────

set -uo pipefail
source "$(dirname "$0")/../deployment/scripts/00_setup_env.sh"

# ── Validate required env vars ────────────────────────────────────────────
: "${GKE_URL:?GKE_URL not set. Export it before running.}"
: "${CLOUDRUN_URL:?CLOUDRUN_URL not set. Export it before running.}"

if [[ -z "${TFIDF_SERVICE_URL:-}" ]]; then
    echo "WARNING: TFIDF_SERVICE_URL not set — Phase C TF-IDF cold-start will be skipped."
fi
if [[ -z "${DISTILBERT_SERVICE_URL:-}" ]]; then
    echo "WARNING: DISTILBERT_SERVICE_URL not set — Phase C DistilBERT cold-start will be skipped."
fi

pip install locust --quiet

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"

# ── Helper: run one Locust scenario ───────────────────────────────────────
run_locust() {
    local class=$1 name=$2 users=$3 rate=$4 duration=$5
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
        2>&1 | tail -5 || true
    echo "  -> ${RESULTS_DIR}/${name}_stats.csv"
}

# ── Helper: log HPA replica counts every 5s in background ────────────────
watch_hpa() {
    local log_file=$1
    echo "timestamp,tfidf_replicas,distilbert_replicas" > "${log_file}"
    while true; do
        local ts tr dr
        ts=$(date +%s)
        tr=$(kubectl get hpa tfidf-hpa -n mh-classifier \
            -o jsonpath='{.status.currentReplicas}' 2>/dev/null || echo "?")
        dr=$(kubectl get hpa distilbert-hpa -n mh-classifier \
            -o jsonpath='{.status.currentReplicas}' 2>/dev/null || echo "?")
        echo "${ts},${tr},${dr}" >> "${log_file}"
        sleep 5
    done
}

# ── Helper: countdown timer ───────────────────────────────────────────────
countdown() {
    local seconds=$1 label=$2
    echo ""
    echo "Waiting ${label}..."
    while [[ $seconds -gt 0 ]]; do
        printf "\r  %3d seconds remaining..." "$seconds"
        sleep 10
        seconds=$((seconds - 10))
    done
    printf "\r  Done.                        \n"
}

# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================="
echo " PHASE A: STEADY TRAFFIC  (1 req/sec, 60s)"
echo " Purpose: warm-path baseline latency, no scaling"
echo "======================================================="

run_locust GkeTfIdf            "gke_tfidf_steady"           1  1 "60s"
run_locust GkeDistilBert       "gke_distilbert_steady"       1  1 "60s"
run_locust CloudRunTfIdf       "cloudrun_tfidf_steady"       1  1 "60s"
run_locust CloudRunDistilBert  "cloudrun_distilbert_steady"  1  1 "60s"

# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================="
echo " PHASE B1: BURST TRAFFIC  (100 users, 10/sec ramp, 100s)"
echo " Purpose: auto-scaling, tail latency, error rate under peak load"
echo " HPA replica count logged every 5s -> results/gke_hpa_burst.log"
echo "======================================================="

HPA_LOG="${RESULTS_DIR}/gke_hpa_burst.log"
watch_hpa "${HPA_LOG}" &
HPA_PID=$!
echo "HPA watcher started (PID=${HPA_PID})"

run_locust GkeTfIdf            "gke_tfidf_burst"            100 10 "100s"
run_locust GkeDistilBert       "gke_distilbert_burst"        100 10 "100s"
run_locust CloudRunTfIdf       "cloudrun_tfidf_burst"        100 10 "100s"
run_locust CloudRunDistilBert  "cloudrun_distilbert_burst"   100 10 "100s"

kill "${HPA_PID}" 2>/dev/null || true
echo "HPA watcher stopped. Log saved: ${HPA_LOG}"

# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================="
echo " PHASE B2: MEDIUM LOAD  (20 users, 2/sec ramp, 120s)"
echo " Purpose: saturation point analysis for DistilBERT"
echo " Identifies where DistilBERT degrades before full saturation"
echo " Comparable to teammate's scaling slope / saturation analysis"
echo "======================================================="

HPA_LOG_MEDIUM="${RESULTS_DIR}/gke_hpa_medium.log"
watch_hpa "${HPA_LOG_MEDIUM}" &
HPA_PID_MEDIUM=$!
echo "HPA watcher started (PID=${HPA_PID_MEDIUM})"

run_locust GkeDistilBert       "gke_distilbert_medium"       20 2 "120s"
run_locust CloudRunDistilBert  "cloudrun_distilbert_medium"  20 2 "120s"

kill "${HPA_PID_MEDIUM}" 2>/dev/null || true
echo "HPA watcher stopped. Log saved: ${HPA_LOG_MEDIUM}"

# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================="
echo " PHASE C: COLD-START PROBES  (Cloud Run only)"
echo " Purpose: isolated per-model cold-start measurement"
echo " Uses dedicated single-model services (not combined service)"
echo " Waiting 15 minutes for instances to go fully idle..."
echo " Press Ctrl+C to skip and go straight to analysis."
echo "======================================================="

countdown 900 "15 minutes for Cloud Run to go cold"

# TF-IDF cold start
if [[ -n "${TFIDF_SERVICE_URL:-}" ]]; then
    echo ""
    echo "── Probing TF-IDF cold start ──"
    locust \
        -f tests/locustfile.py ColdStartProbeTfIdf \
        --headless -u 1 -r 1 --run-time 60s \
        --csv  "${RESULTS_DIR}/cloudrun_tfidf_true_coldstart" \
        --html "${RESULTS_DIR}/cloudrun_tfidf_true_coldstart.html" \
        --logfile "${RESULTS_DIR}/cloudrun_tfidf_true_coldstart.log" \
        2>&1 | tail -5
    echo "  -> ${RESULTS_DIR}/cloudrun_tfidf_true_coldstart_stats.csv"
    grep "COLD START" "${RESULTS_DIR}/cloudrun_tfidf_true_coldstart.log" || true
else
    echo "Skipping — TFIDF_SERVICE_URL not set."
fi

countdown 300 "5 minutes for DistilBERT service to go cold"

# DistilBERT cold start
if [[ -n "${DISTILBERT_SERVICE_URL:-}" ]]; then
    echo ""
    echo "── Probing DistilBERT cold start ──"
    locust \
        -f tests/locustfile.py ColdStartProbeDistilBert \
        --headless -u 1 -r 1 --run-time 300s \
        --csv  "${RESULTS_DIR}/cloudrun_distilbert_true_coldstart" \
        --html "${RESULTS_DIR}/cloudrun_distilbert_true_coldstart.html" \
        --logfile "${RESULTS_DIR}/cloudrun_distilbert_true_coldstart.log" \
        2>&1 | tail -5
    echo "  -> ${RESULTS_DIR}/cloudrun_distilbert_true_coldstart_stats.csv"
    grep "COLD START" "${RESULTS_DIR}/cloudrun_distilbert_true_coldstart.log" || true
else
    echo "Skipping — DISTILBERT_SERVICE_URL not set."
fi

# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "======================================================="
echo " RESULTS SUMMARY"
echo "======================================================="

python3 - <<'EOF'
import csv, glob, os

files = sorted(glob.glob("results/*_stats.csv"))
print(f"\n{'Config':<46} {'p50':>6} {'p95':>6} {'p99':>7} {'RPS':>7} {'Err%':>7}")
print("-" * 78)
for f in files:
    name = os.path.basename(f).replace("_stats.csv", "")
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row.get("Name") == "Aggregated":
                n = int(row.get("Request Count", 0))
                e = int(row.get("Failure Count", 0))
                err = f"{e/n*100:.2f}" if n else "N/A"
                print(
                    f"{name:<46} "
                    f"{float(row.get('50%',  0)):>6.0f} "
                    f"{float(row.get('95%',  0)):>6.0f} "
                    f"{float(row.get('99%',  0)):>7.0f} "
                    f"{float(row.get('Requests/s', 0)):>7.1f} "
                    f"{err:>7}"
                )
EOF

echo ""
echo "======================================================="
echo " OUTPUT FILES"
echo "======================================================="
echo "  Results        : ${RESULTS_DIR}/"
echo "  HTML reports   : ${RESULTS_DIR}/*.html"
echo "  HPA burst log  : ${RESULTS_DIR}/gke_hpa_burst.log"
echo "  HPA medium log : ${RESULTS_DIR}/gke_hpa_medium.log"
echo ""
echo "Next: python3 tests/06_analyze_results.py"
echo "      python3 tests/07_plot_results.py"
echo ""
