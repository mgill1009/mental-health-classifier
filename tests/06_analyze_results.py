"""
06_analyze_results.py
Parses Locust CSV results and GKE HPA log, produces:
  results/summary_table.csv      -- all 10 configs in one comparison table
  results/scaling_analysis.json  -- GKE scale-up latency from HPA log
  results/cost_estimate.csv      -- estimated monthly cost per platform
  Printed findings mapped to each professor evaluation question

Usage:
  python3 tests/06_analyze_results.py

Requires: results/ directory populated by 05_run_load_tests.sh
"""

import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from xml.etree.ElementPath import find

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Parse Locust stats CSV ────────────────────────────────────────────────
def parse_stats(path: str) -> dict:
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("Name") == "Aggregated":
                n    = int(row.get("Request Count", 0))
                fail = int(row.get("Failure Count", 0))
                return {
                    "total_requests": n,
                    "failure_count":  fail,
                    "error_pct":      round(fail / n * 100, 2) if n else None,
                    "p50_ms":         float(row.get("50%", 0)),
                    "p95_ms":         float(row.get("95%", 0)),
                    "p99_ms":         float(row.get("99%", 0)),
                    "rps":            float(row.get("Requests/s", 0)),
                    "min_ms":         float(row.get("Min Response Time", 0)),
                    "max_ms":         float(row.get("Max Response Time", 0)),
                }
    return {}


# ── Parse GKE HPA log ─────────────────────────────────────────────────────
def parse_hpa_log(path: str) -> dict:
    """
    Compute scale-up latency from the HPA watcher log.

    Scale-up latency definition:
      Time (seconds) from the first moment REPLICAS increases above baseline
      until REPLICAS reaches its peak value and stops changing.

    This is the concrete metric that answers the professor's auto-scaling
    question. It captures: how long did the cluster take to respond to
    the burst, and how does that differ between TF-IDF (fast readiness probe)
    and DistilBERT (45s readiness probe)?
    """
    if not os.path.exists(path):
        return {"error": f"HPA log not found at {path}"}

    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "ts": int(row["timestamp"]),
                    "tfidf":      int(row["tfidf_replicas"]),
                    "distilbert": int(row["distilbert_replicas"]),
                })
            except (ValueError, KeyError):
                continue

    if len(rows) < 2:
        return {"error": "Insufficient HPA data points"}

    def scale_latency(col):
        baseline   = rows[0][col]
        peak       = baseline
        start_ts   = None
        peak_ts    = None
        for row in rows:
            if row[col] > baseline and start_ts is None:
                start_ts = row["ts"]
            if row[col] > peak:
                peak    = row[col]
                peak_ts = row["ts"]
        if start_ts and peak_ts:
            return {
                "baseline_replicas":  baseline,
                "peak_replicas":      peak,
                "scale_up_latency_s": peak_ts - start_ts,
                "note": (
                    f"Scaled from {baseline} to {peak} pods "
                    f"in {peak_ts - start_ts}s"
                ),
            }
        return {"error": f"No scaling observed for {col}"}

    return {
        "tfidf":      scale_latency("tfidf"),
        "distilbert": scale_latency("distilbert"),
    }


# ── Cost estimates ────────────────────────────────────────────────────────
def cost_estimates(rps: float = 1.0) -> list:
    """
    Approximate monthly cost for each platform at a given request rate.
    Based on us-central1 pricing (2025).

    GKE Standard (e2-standard-4, 2 nodes):
      Fixed cost regardless of traffic.
      Becomes cost-efficient above ~200 req/sec sustained.

    Cloud Run:
      Scales to zero. Cost proportional to traffic.
      $0.000024/vCPU-second, 2 vCPU, ~0.1s average request duration.
      Becomes cost-efficient below ~200 req/sec sustained.

    Break-even point (where costs are equal) is a key project finding.
    """
    reqs_per_month = rps * 60 * 60 * 24 * 30

    gke_cost = 2 * 0.134 * 24 * 30   # 2 nodes * $0.134/hr * 720 hrs

    # Cloud Run: per-vCPU-second billing
    cloudrun_cost = reqs_per_month * 0.000024 * 2 * 0.1

    # Break-even: where cloudrun_cost == gke_cost
    # cloudrun_cost = rps * 60*60*24*30 * 0.000024 * 2 * 0.1
    # gke_cost      = 2 * 0.134 * 720
    # rps_breakeven = gke_cost / (60*60*24*30 * 0.000024 * 2 * 0.1)
    rps_breakeven = gke_cost / (60 * 60 * 24 * 30 * 0.000024 * 2 * 0.1)

    return [
        {
            "platform":          "GKE (2x e2-standard-4)",
            "rps":               rps,
            "est_monthly_usd":   round(gke_cost, 2),
            "cost_model":        "Fixed — always-on nodes",
            "breakeven_rps":     round(rps_breakeven, 1),
        },
        {
            "platform":          "Cloud Run",
            "rps":               rps,
            "est_monthly_usd":   round(cloudrun_cost, 2),
            "cost_model":        "Variable — pay per request",
            "breakeven_rps":     round(rps_breakeven, 1),
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 70)
    print(" MENTAL HEALTH CLASSIFIER — RESULTS ANALYSIS")
    print(f" GKE (containerized) vs Cloud Run (serverless)")
    print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ── Collect stats files ───────────────────────────────────────────────
    stats_files = sorted(glob.glob(str(RESULTS_DIR / "*_stats.csv")))
    if not stats_files:
        print("\nNo results CSVs found. Run 05_run_load_tests.sh first.")
        return

    rows = []
    for path in stats_files:
        name  = Path(path).stem.replace("_stats", "")
        parts = name.split("_")
        data  = parse_stats(path)
        data.update({
            "config":   name,
            "platform": parts[0],                              # gke / cloudrun
            "model":    parts[1] if len(parts) > 1 else "?",  # tfidf / distilbert
            "scenario": parts[2] if len(parts) > 2 else "?",  # steady / burst / coldstart
        })
        rows.append(data)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'Config':<42} {'p50':>5} {'p95':>5} {'p99':>5} {'RPS':>6} {'Err%':>6}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['config']:<42} "
            f"{r.get('p50_ms', 0):>5.0f} "
            f"{r.get('p95_ms', 0):>5.0f} "
            f"{r.get('p99_ms', 0):>5.0f} "
            f"{r.get('rps', 0):>6.1f} "
            f"{str(r.get('error_pct', 'N/A')):>6}"
        )

    summary_path = RESULTS_DIR / "summary_table.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {summary_path}")

    # ── GKE auto-scaling analysis ─────────────────────────────────────────
    print()
    print("=" * 70)
    print(" GKE AUTO-SCALING ANALYSIS")
    print(" Metric: seconds from first replica increase until peak replicas")
    print(" This directly answers the professor's auto-scaling metric question.")
    print("=" * 70)

    hpa_data = parse_hpa_log(str(RESULTS_DIR / "gke_hpa_burst.log"))
    if "error" not in hpa_data:
        for model, result in hpa_data.items():
            print(f"\n  {model.upper()}:")
            if "error" not in result:
                print(f"    Baseline replicas  : {result['baseline_replicas']}")
                print(f"    Peak replicas      : {result['peak_replicas']}")
                print(f"    Scale-up latency   : {result['scale_up_latency_s']}s")
                print(f"    Note               : {result['note']}")
            else:
                print(f"    {result['error']}")
    else:
        print(f"\n  {hpa_data['error']}")

    with open(RESULTS_DIR / "scaling_analysis.json", "w") as f:
        json.dump(hpa_data, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'scaling_analysis.json'}")

    # ── Cost estimates ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(" COST ESTIMATES (us-central1, 2025 pricing)")
    print("=" * 70)

    for rps in [1, 10, 50, 200]:
        estimates = cost_estimates(rps)
        print(f"\n  At {rps} req/sec:")
        for e in estimates:
            print(f"    {e['platform']:<30} ${e['est_monthly_usd']:>8.2f}/month")
        print(f"    Break-even: {estimates[0]['breakeven_rps']} req/sec")

    with open(RESULTS_DIR / "cost_estimate.csv", "w", newline="") as f:
        flat = []
        for rps in [1, 10, 50, 200]:
            for e in cost_estimates(rps):
                flat.append(e)
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    print(f"\n  Saved: {RESULTS_DIR / 'cost_estimate.csv'}")

    # ── Key findings mapped to evaluation questions ───────────────────────
    def find(platform, model, scenario):
        return next(
            (r for r in rows
             if r["platform"] == platform
             and r["model"] == model
             and r["scenario"] == scenario),
            {}
        )

    print()
    print("=" * 70)
    print(" KEY FINDINGS (mapped to professor evaluation questions)")
    print("=" * 70)

    print("\nQ1: Does deployment architecture affect warm-path latency? (steady traffic)")
    for model in ["tfidf", "distilbert"]:
        gke = find("gke", model, "steady")
        cr  = find("cloudrun", model, "steady")
        print(f"\n  {model.upper()} p99 (ms):")
        print(f"    GKE       : {gke.get('p99_ms', 'N/A')}")
        print(f"    Cloud Run : {cr.get('p99_ms',  'N/A')}")

    print("\nQ2: Does model size amplify Cloud Run cold-start penalty? (H1)")
    cs_tfidf = find("cloudrun", "tfidf",      "true")
    cs_bert  = find("cloudrun", "distilbert", "true")
    t_p99 = cs_tfidf.get("p99_ms")
    b_p99 = cs_bert.get("p99_ms")
    print(f"  TF-IDF cold start p99    : {t_p99} ms")
    print(f"  DistilBERT cold start p99: {b_p99} ms")
    if t_p99 and b_p99 and t_p99 > 0:
        ratio = b_p99 / t_p99
        verdict = "CONFIRMED" if ratio >= 5 else "NOT confirmed"
        print(f"  Ratio (H1 expects 5-10x) : {ratio:.1f}x  -> H1 {verdict}")

    print("\nQ3: Error rate under burst traffic?")
    for platform in ["gke", "cloudrun"]:
        for model in ["tfidf", "distilbert"]:
            r = find(platform, model, "burst")
            err = r.get("error_pct", "N/A")
            p99 = r.get("p99_ms", "N/A")
            print(f"  {platform} {model:<12} p99={p99}ms  errors={err}%")

    print("\nQ4: Auto-scaling (GKE) -- see scaling_analysis.json for full detail")
    if "error" not in hpa_data:
        for model, result in hpa_data.items():
            if "error" not in result:
                print(
                    f"  {model:<12} scaled "
                    f"{result['baseline_replicas']}->{result['peak_replicas']} pods "
                    f"in {result['scale_up_latency_s']}s"
                )

    print()
    print("  All results saved to results/")
    print("  HTML reports: results/*.html")


if __name__ == "__main__":
    main()
