"""
07_plot_results.py
Reads Locust CSV output and produces report-ready charts.

Usage:
  pip install matplotlib pandas numpy
  python3 tests/07_plot_results.py

Outputs (saved to results/):
  plot_01_latency_comparison.png   -- p50/p99 bar chart across all configs
  plot_02_cold_start.png           -- cold-start overhead GKE vs Cloud Run
  plot_03_error_rate.png           -- error rate under burst traffic
  plot_04_cost_curve.png           -- monthly cost vs request rate
  plot_05_hpa_scaling.png          -- GKE replica count over time (burst test)
"""

import os
import csv
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR     = RESULTS_DIR

# ── Colours (matches project design system) ───────────────────────────────
GKE_COLOR       = "#1D9E75"   # teal
CLOUDRUN_COLOR  = "#7F77DD"   # purple
TFIDF_COLOR     = "#378ADD"   # blue
DISTILBERT_COLOR= "#D85A30"   # coral
GRAY            = "#888780"
LIGHT_GRAY      = "#F1EFE8"
BLACK           = "#2C2C2A"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#E0DEDB",
    "grid.linewidth":     0.6,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# ── Helpers ───────────────────────────────────────────────────────────────
def load_stats(prefix):
    """Load aggregated row from a Locust _stats.csv file."""
    path = RESULTS_DIR / f"{prefix}_stats.csv"
    if not path.exists():
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("Name") == "Aggregated":
                n    = int(row.get("Request Count", 0))
                fail = int(row.get("Failure Count", 0))
                return {
                    "p50":       float(row.get("50%",  0)),
                    "p95":       float(row.get("95%",  0)),
                    "p99":       float(row.get("99%",  0)),
                    "rps":       float(row.get("Requests/s", 0)),
                    "error_pct": round(fail / n * 100, 2) if n else 0,
                    "n":         n,
                }
    return None


def label_bars(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.02,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=9, color=BLACK,
            )


# ── Plot 1: Latency comparison (steady traffic) ───────────────────────────
def plot_latency_comparison():
    configs = [
        ("gke_tfidf_steady",          "GKE\nTF-IDF",      GKE_COLOR,      "solid"),
        ("gke_distilbert_steady",     "GKE\nDistilBERT",   GKE_COLOR,      "solid"),
        ("cloudrun_tfidf_steady",     "Cloud Run\nTF-IDF", CLOUDRUN_COLOR, "solid"),
        ("cloudrun_distilbert_steady","Cloud Run\nDistilBERT", CLOUDRUN_COLOR, "solid"),
    ]

    labels, p50s, p99s, colors = [], [], [], []
    for prefix, label, color, _ in configs:
        s = load_stats(prefix)
        if s:
            labels.append(label)
            p50s.append(s["p50"])
            p99s.append(s["p99"])
            colors.append(color)

    if not labels:
        print("No steady data found for latency comparison plot.")
        return

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - w/2, p50s, w, label="p50 (median)", color=colors, alpha=0.9)
    b2 = ax.bar(x + w/2, p99s, w, label="p99 (tail)",   color=colors, alpha=0.45)

    label_bars(ax, b1)
    label_bars(ax, b2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Steady traffic latency — GKE vs Cloud Run", fontweight="bold", pad=12)

    gke_patch = mpatches.Patch(color=GKE_COLOR,      label="GKE")
    cr_patch  = mpatches.Patch(color=CLOUDRUN_COLOR, label="Cloud Run")
    p50_patch = mpatches.Patch(color="gray", alpha=0.9,  label="p50 (median)")
    p99_patch = mpatches.Patch(color="gray", alpha=0.45, label="p99 (tail)")
    ax.legend(handles=[gke_patch, cr_patch, p50_patch, p99_patch],
              loc="upper left", framealpha=0.9)

    out = OUT_DIR / "plot_01_latency_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: Cold start — first vs warm requests ───────────────────────────
def plot_cold_start():
    """
    Reads stats_history CSVs to extract first-request latency (cold)
    vs median warm latency for Cloud Run configs.
    """
    configs = [
        ("cloudrun_tfidf_steady",      "Cloud Run TF-IDF",      TFIDF_COLOR),
        ("cloudrun_distilbert_steady", "Cloud Run DistilBERT",  DISTILBERT_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for prefix, label, color in configs:
        hist_path = RESULTS_DIR / f"{prefix}_stats_history.csv"
        stats     = load_stats(prefix)
        if not hist_path.exists() or not stats:
            continue

        # Read history: find max latency (cold) vs steady median
        times, medians = [], []
        with open(hist_path) as f:
            for row in csv.DictReader(f):
                try:
                    t = float(row.get("Timestamp", 0))
                    m = float(row.get("50%", 0))
                    if m > 0:
                        times.append(t)
                        medians.append(m)
                except ValueError:
                    continue

        if not times:
            continue

        # Normalise timestamps to seconds from start
        t0 = times[0]
        times = [t - t0 for t in times]

        ax.plot(times, medians, color=color, linewidth=2, label=label)
        plotted = True

    if not plotted:
        # Fallback: bar chart from stats CSV
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        labels, warm, cold = [], [], []
        for prefix, label, color in configs:
            s = load_stats(prefix)
            if s:
                labels.append(label)
                warm.append(s["p50"])
                cold.append(s["p99"])  # p99 captures the cold-start spike

        if not labels:
            print("No Cloud Run data for cold-start plot.")
            plt.close(fig)
            return

        x = np.arange(len(labels))
        w = 0.35
        b1 = ax.bar(x - w/2, warm, w, label="p50 warm (ms)",       color=[TFIDF_COLOR, DISTILBERT_COLOR], alpha=0.9)
        b2 = ax.bar(x + w/2, cold, w, label="p99 cold start (ms)", color=[TFIDF_COLOR, DISTILBERT_COLOR], alpha=0.45)
        label_bars(ax, b1)
        label_bars(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Latency (ms)")
        ax.legend()
    else:
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Median latency (ms)")
        ax.legend()

    ax.set_title("Cloud Run cold-start vs warm latency", fontweight="bold", pad=12)

    # Annotate the cold-start region
    ax.axvspan(0, 5, alpha=0.08, color="red", label="Cold-start window")
    ax.text(2.5, ax.get_ylim()[1] * 0.92, "cold\nstart",
            ha="center", fontsize=9, color="red", alpha=0.7)

    out = OUT_DIR / "plot_02_cold_start.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 3: Error rate under burst traffic ────────────────────────────────
def plot_error_rate():
    configs = [
        ("gke_tfidf_burst",           "GKE\nTF-IDF",           GKE_COLOR),
        ("gke_distilbert_burst",      "GKE\nDistilBERT",        GKE_COLOR),
        ("cloudrun_tfidf_burst",      "Cloud Run\nTF-IDF",      CLOUDRUN_COLOR),
        ("cloudrun_distilbert_burst", "Cloud Run\nDistilBERT",  CLOUDRUN_COLOR),
    ]

    labels, errors, p99s = [], [], []
    for prefix, label, color in configs:
        s = load_stats(prefix)
        if s:
            labels.append(label)
            errors.append(s["error_pct"])
            p99s.append(s["p99"])

    if not labels:
        print("No burst data found for error rate plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [GKE_COLOR, GKE_COLOR, CLOUDRUN_COLOR, CLOUDRUN_COLOR][:len(labels)]

    # Error rate
    bars1 = ax1.bar(labels, errors, color=colors, alpha=0.85)
    label_bars(ax1, bars1, fmt="{:.1f}%")
    ax1.set_ylabel("Error rate (%)")
    ax1.set_title("Error rate — burst traffic (100 users)", fontweight="bold", pad=12)
    ax1.set_ylim(0, max(errors + [5]) * 1.2)

    # p99 under burst
    bars2 = ax2.bar(labels, p99s, color=colors, alpha=0.85)
    label_bars(ax2, bars2)
    ax2.set_ylabel("p99 latency (ms)")
    ax2.set_title("p99 latency — burst traffic (100 users)", fontweight="bold", pad=12)

    gke_patch = mpatches.Patch(color=GKE_COLOR,      label="GKE")
    cr_patch  = mpatches.Patch(color=CLOUDRUN_COLOR, label="Cloud Run")
    fig.legend(handles=[gke_patch, cr_patch], loc="upper right",
               bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "plot_03_burst_performance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 4: Cost curve ────────────────────────────────────────────────────
def plot_cost_curve():
    rps_range = np.linspace(0.1, 300, 500)

    # GKE: fixed cost (2 x e2-standard-4)
    gke_monthly = np.full_like(rps_range, 2 * 0.134 * 24 * 30)

    # Cloud Run: per-vCPU-second
    cloudrun_monthly = rps_range * 60 * 60 * 24 * 30 * 0.000024 * 2 * 0.1

    # Break-even
    breakeven = gke_monthly[0] / (60 * 60 * 24 * 30 * 0.000024 * 2 * 0.1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rps_range, gke_monthly,      color=GKE_COLOR,      linewidth=2.5, label="GKE (fixed cost)")
    ax.plot(rps_range, cloudrun_monthly, color=CLOUDRUN_COLOR, linewidth=2.5, label="Cloud Run (pay-per-request)")
    ax.axvline(breakeven, color=GRAY, linestyle="--", linewidth=1.5)
    ax.text(breakeven + 3, gke_monthly[0] * 0.5,
            f"Break-even\n~{breakeven:.0f} req/sec",
            color=GRAY, fontsize=10)

    # Shade regions
    ax.fill_between(rps_range, gke_monthly, cloudrun_monthly,
                    where=cloudrun_monthly < gke_monthly,
                    alpha=0.08, color=CLOUDRUN_COLOR, label="Cloud Run cheaper")
    ax.fill_between(rps_range, gke_monthly, cloudrun_monthly,
                    where=cloudrun_monthly >= gke_monthly,
                    alpha=0.08, color=GKE_COLOR, label="GKE cheaper")

    ax.set_xlabel("Sustained request rate (req/sec)")
    ax.set_ylabel("Estimated monthly cost (USD)")
    ax.set_title("Cost comparison — GKE vs Cloud Run", fontweight="bold", pad=12)
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, max(cloudrun_monthly[-1], gke_monthly[0]) * 1.1)

    out = OUT_DIR / "plot_04_cost_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 5: GKE HPA scaling over time ────────────────────────────────────
def plot_hpa_scaling():
    hpa_path = RESULTS_DIR / "gke_hpa_burst.log"
    if not hpa_path.exists():
        print("No HPA log found — skipping scaling plot.")
        print("To capture HPA data, run: kubectl get hpa -n mh-classifier -w")
        print("during the burst test and save output to results/gke_hpa_burst.log")
        return

    times_tfidf, reps_tfidf = [], []
    times_bert,  reps_bert  = [], []

    with open(hpa_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["timestamp"])
                tr = int(row["tfidf_replicas"])
                dr = int(row["distilbert_replicas"])
                times_tfidf.append(ts)
                reps_tfidf.append(tr)
                times_bert.append(ts)
                reps_bert.append(dr)
            except (ValueError, KeyError):
                continue

    if not times_tfidf:
        print("HPA log exists but has no parseable data.")
        return

    t0 = times_tfidf[0]
    times_tfidf = [t - t0 for t in times_tfidf]
    times_bert  = [t - t0 for t in times_bert]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(times_tfidf, reps_tfidf, color=TFIDF_COLOR,      linewidth=2.5,
            where="post", label="TF-IDF replicas")
    ax.step(times_bert,  reps_bert,  color=DISTILBERT_COLOR, linewidth=2.5,
            where="post", label="DistilBERT replicas")

    ax.set_xlabel("Time from burst start (seconds)")
    ax.set_ylabel("Pod replica count")
    ax.set_title("GKE auto-scaling during burst traffic (100 users)", fontweight="bold", pad=12)
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(max(reps_tfidf), max(reps_bert)) + 1)

    # Annotate scale-up events
    for i in range(1, len(reps_tfidf)):
        if reps_tfidf[i] > reps_tfidf[i-1]:
            ax.annotate(f"+{reps_tfidf[i]-reps_tfidf[i-1]}",
                       xy=(times_tfidf[i], reps_tfidf[i]),
                       xytext=(times_tfidf[i]+2, reps_tfidf[i]+0.2),
                       fontsize=9, color=TFIDF_COLOR)

    out = OUT_DIR / "plot_05_hpa_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 6: Model comparison summary ─────────────────────────────────────
def plot_summary_comparison():
    """Side-by-side: TF-IDF vs DistilBERT across both platforms."""
    scenarios = [
        ("gke_tfidf_steady",           "GKE steady",  "TF-IDF",     GKE_COLOR,      TFIDF_COLOR),
        ("gke_distilbert_steady",      "GKE steady",  "DistilBERT", GKE_COLOR,       DISTILBERT_COLOR),
        ("cloudrun_tfidf_steady",      "Cloud Run steady", "TF-IDF", CLOUDRUN_COLOR, TFIDF_COLOR),
        ("cloudrun_distilbert_steady", "Cloud Run steady", "DistilBERT", CLOUDRUN_COLOR, DISTILBERT_COLOR),
        ("gke_tfidf_burst",            "GKE burst",   "TF-IDF",     GKE_COLOR,      TFIDF_COLOR),
        ("gke_distilbert_burst",       "GKE burst",   "DistilBERT", GKE_COLOR,       DISTILBERT_COLOR),
        ("cloudrun_tfidf_burst",       "Cloud Run burst", "TF-IDF", CLOUDRUN_COLOR, TFIDF_COLOR),
        ("cloudrun_distilbert_burst",  "Cloud Run burst", "DistilBERT", CLOUDRUN_COLOR, DISTILBERT_COLOR),
    ]

    rows = []
    for prefix, scenario, model, platform_color, model_color in scenarios:
        s = load_stats(prefix)
        if s:
            rows.append({
                "label":    f"{scenario}\n{model}",
                "p50":      s["p50"],
                "p99":      s["p99"],
                "err":      s["error_pct"],
                "color":    platform_color,
                "mcolor":   model_color,
            })

    if not rows:
        print("Not enough data for summary comparison.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))

    # p50
    axes[0].bar(x, [r["p50"] for r in rows], color=[r["color"] for r in rows], alpha=0.85)
    axes[0].set_title("p50 latency (ms)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    axes[0].set_ylabel("ms")

    # p99
    axes[1].bar(x, [r["p99"] for r in rows], color=[r["color"] for r in rows], alpha=0.85)
    axes[1].set_title("p99 latency (ms)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    axes[1].set_ylabel("ms")

    # Error rate
    axes[2].bar(x, [r["err"] for r in rows], color=[r["color"] for r in rows], alpha=0.85)
    axes[2].set_title("Error rate (%)", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    axes[2].set_ylabel("%")

    gke_patch = mpatches.Patch(color=GKE_COLOR,      label="GKE")
    cr_patch  = mpatches.Patch(color=CLOUDRUN_COLOR, label="Cloud Run")
    fig.legend(handles=[gke_patch, cr_patch], loc="upper right", framealpha=0.9)
    fig.suptitle("Full results summary — all configurations", fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUT_DIR / "plot_06_full_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Run all plots ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating plots from {RESULTS_DIR}/\n")
    plot_latency_comparison()
    plot_cold_start()
    plot_error_rate()
    plot_cost_curve()
    plot_hpa_scaling()
    plot_summary_comparison()
    print("\nDone. All plots saved to results/")
