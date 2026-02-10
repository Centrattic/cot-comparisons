#!/usr/bin/env python3
"""
Grouped bar chart comparing entropy probe vs LLM monitor vs predict-mean baseline.

Usage:
    python -m src2.runs.plot_entropy_probe_vs_monitor
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROBE_RESULTS = PROJECT_ROOT / "data" / "forced_response" / "entropy_probe" / "latest" / "results.json"
MONITOR_RESULTS = PROJECT_ROOT / "data" / "forced_response" / "monitor_eval" / "forcing_monitor_results.json"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "forced_response"


def main():
    with open(PROBE_RESULTS) as f:
        probe = json.load(f)
    with open(MONITOR_RESULTS) as f:
        monitor = json.load(f)

    probe_eval = probe["eval_metrics"]
    monitor_eval = monitor["metrics"]

    # ── Data ──────────────────────────────────────────────────────────
    methods = ["Predict Mean\n(baseline)", "LLM Monitor\n(GPT-5.2)", "Entropy Probe\n(layer 32)"]

    metrics = {
        "R²":                    [0.0, monitor_eval["r2"], probe_eval["r2"]],
        "MSE (lower is better)": [probe_eval["baseline_mse"], monitor_eval["mse"], probe_eval["mse"]],
        "Pearson r":             [0.0, monitor_eval["pearson_r"], probe_eval["pearson_r"]],
    }
    metric_colors = {"R²": "#5c6bc0", "MSE (lower is better)": "#ef5350", "Pearson r": "#43a047"}

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(methods))
    n_metrics = len(metrics)
    bar_width = 0.22
    offsets = np.linspace(-(n_metrics - 1) / 2 * bar_width,
                          (n_metrics - 1) / 2 * bar_width, n_metrics)

    for offset, (metric_name, vals) in zip(offsets, metrics.items()):
        bars = ax.bar(x + offset, vals, bar_width,
                      label=metric_name, color=metric_colors[metric_name],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(-0.02, 0.78)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("Predicting Forced Answer Entropy\nProbe  vs  Monitor  vs  Baseline",
                 fontsize=13, pad=14)
    ax.legend(fontsize=10, loc="upper left")

    fig.text(0.5, -0.02,
             f"Eval set: {probe_eval['n_samples']} samples (probe), "
             f"{monitor_eval['n_samples']} samples (monitor) | "
             f"12 held-out questions | R² & Pearson r: higher is better, MSE: lower is better",
             ha="center", fontsize=8.5, color="#666666")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "entropy_probe_vs_monitor.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
