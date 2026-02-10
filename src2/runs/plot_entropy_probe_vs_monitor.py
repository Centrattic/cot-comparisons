#!/usr/bin/env python3
"""
Horizontal bar chart: R² for predicting forced answer entropy.

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
LOGIT_LENS_RESULTS = PROJECT_ROOT / "data" / "forced_response" / "monitor_eval" / "logit_lens_results.json"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "forced_response"


def main():
    with open(PROBE_RESULTS) as f:
        probe = json.load(f)
    with open(MONITOR_RESULTS) as f:
        monitor = json.load(f)

    probe_eval = probe["eval_metrics"]
    monitor_eval = monitor["metrics"]

    has_logit_lens = LOGIT_LENS_RESULTS.exists()
    if has_logit_lens:
        with open(LOGIT_LENS_RESULTS) as f:
            logit_lens_eval = json.load(f)["metrics"]
    else:
        print(f"Warning: logit lens results not found at {LOGIT_LENS_RESULTS}")
        print("  Run: python -m src2.runs.run_logit_lens_baseline")

    # ── Data (ordered worst → best, bottom → top) ────────────────────
    methods = []
    r2_vals = []
    r_vals = []
    colors = []

    if has_logit_lens:
        methods.append("Logit Lens (layer 32)")
        r2_vals.append(logit_lens_eval["r2"])
        r_vals.append(logit_lens_eval["pearson_r"])
        colors.append("#bdbdbd")

    methods += ["Predict Mean", "LLM Monitor (GPT-5.2)", "Entropy Probe (layer 32)"]
    r2_vals += [0.0, monitor_eval["r2"], probe_eval["r2"]]
    r_vals += [0.0, monitor_eval["pearson_r"], probe_eval["pearson_r"]]
    colors += ["#e0e0e0", "#78909c", "#5c6bc0"]

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 2.8))

    y = np.arange(len(methods))
    bars = ax.barh(y, r2_vals, height=0.55, color=colors, edgecolor="white", linewidth=0.8)

    # Labels on bars
    for bar, r2, r in zip(bars, r2_vals, r_vals):
        w = bar.get_width()
        if r2 < 0:
            # Negative bars: label to the right of zero
            ax.text(0.02, bar.get_y() + bar.get_height() / 2,
                    f"R² = {r2:.2f}   r = {r:.2f}",
                    ha="left", va="center", fontsize=9, color="#666")
        else:
            ax.text(max(w, 0) + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"R² = {r2:.2f}   r = {r:.2f}",
                    ha="left", va="center", fontsize=9,
                    color="#333", fontweight="medium")

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("R²  (higher is better)", fontsize=10)
    ax.axvline(x=0, color="#333", linewidth=0.6)
    ax.set_xlim(min(min(r2_vals) - 0.15, -0.15), max(r2_vals) + 0.28)
    ax.set_title("Predicting Forced Answer Entropy (eval set, 12 held-out questions)",
                 fontsize=11, pad=10)

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "entropy_probe_vs_monitor.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
