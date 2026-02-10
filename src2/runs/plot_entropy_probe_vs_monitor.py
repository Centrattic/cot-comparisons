#!/usr/bin/env python3
"""
Bar chart comparing entropy probe vs LLM monitor vs predict-mean baseline.

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
    colors = ["#9e9e9e", "#5c6bc0", "#43a047"]

    r2_vals = [0.0, monitor_eval["r2"], probe_eval["r2"]]
    mse_vals = [probe_eval["baseline_mse"], monitor_eval["mse"], probe_eval["mse"]]
    pearson_vals = [0.0, monitor_eval["pearson_r"], probe_eval["pearson_r"]]

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Predicting Forced Answer Entropy: Probe vs Monitor vs Baseline", fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(methods))
    bar_width = 0.55

    # R²
    ax = axes[0]
    bars = ax.bar(x, r2_vals, bar_width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("R² (higher is better)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(-0.05, 0.55)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # MSE
    ax = axes[1]
    bars = ax.bar(x, mse_vals, bar_width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("MSE (lower is better)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 0.25)
    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Pearson r
    ax = axes[2]
    bars = ax.bar(x, pearson_vals, bar_width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("Pearson r (higher is better)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(-0.05, 0.85)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    for bar, val in zip(bars, pearson_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotation with sample counts
    fig.text(0.5, -0.04,
             f"Eval set: {probe_eval['n_samples']} samples (probe), "
             f"{monitor_eval['n_samples']} samples (monitor) | "
             f"12 held-out questions | Target: Shannon entropy of forced answer distribution",
             ha="center", fontsize=9, color="#666666")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "entropy_probe_vs_monitor.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
