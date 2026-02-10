#!/usr/bin/env python3
"""
Horizontal bar charts: R² and MSE for predicting forced answer entropy.

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
    mse_vals = []
    colors = []

    if has_logit_lens:
        methods.append("Logit Lens (layer 32)")
        r2_vals.append(logit_lens_eval["r2"])
        mse_vals.append(logit_lens_eval["mse"])
        colors.append("#bdbdbd")

    methods += ["Predict Mean", "LLM Monitor (GPT-5.2)", "Entropy Probe (layer 32)"]
    r2_vals += [0.0, monitor_eval["r2"], probe_eval["r2"]]
    mse_vals += [probe_eval["baseline_mse"], monitor_eval["mse"], probe_eval["mse"]]
    colors += ["#e0e0e0", "#78909c", "#5c6bc0"]

    # ── Figure: two panels ────────────────────────────────────────────
    fig, (ax_r2, ax_mse) = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)

    y = np.arange(len(methods))
    bar_h = 0.55

    # ── Left panel: R² (higher is better) ─────────────────────────────
    bars_r2 = ax_r2.barh(y, r2_vals, height=bar_h, color=colors,
                         edgecolor="white", linewidth=0.8)
    for bar, r2 in zip(bars_r2, r2_vals):
        x_pos = max(bar.get_width(), 0) + 0.02 if r2 >= 0 else 0.02
        ax_r2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                   f"{r2:.2f}", ha="left", va="center", fontsize=9.5,
                   color="#666" if r2 < 0 else "#333", fontweight="medium")

    ax_r2.set_yticks(y)
    ax_r2.set_yticklabels(methods, fontsize=10)
    ax_r2.set_xlabel("R²  (higher is better)", fontsize=10)
    ax_r2.axvline(x=0, color="#333", linewidth=0.6)
    ax_r2.set_xlim(min(min(r2_vals) - 0.12, -0.12), max(r2_vals) + 0.18)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)
    ax_r2.tick_params(left=False)

    # ── Right panel: MSE (lower is better) ────────────────────────────
    bars_mse = ax_mse.barh(y, mse_vals, height=bar_h, color=colors,
                           edgecolor="white", linewidth=0.8)
    for bar, mse in zip(bars_mse, mse_vals):
        ax_mse.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{mse:.3f}", ha="left", va="center", fontsize=9.5,
                    color="#333", fontweight="medium")

    ax_mse.set_xlabel("MSE  (lower is better)", fontsize=10)
    ax_mse.set_xlim(0, max(mse_vals) + 0.09)
    ax_mse.spines["top"].set_visible(False)
    ax_mse.spines["right"].set_visible(False)
    ax_mse.tick_params(left=False)

    fig.suptitle("Predicting Forced Answer Entropy (eval set, 12 held-out questions)",
                 fontsize=11, y=1.02)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "entropy_probe_vs_monitor.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
