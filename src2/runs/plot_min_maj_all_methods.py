#!/usr/bin/env python3
"""
Plot comparing all majority/minority answer classification methods.

Methods (all using LOO-CV across 7 prompts):
  1. LLM Monitor (GPT-4.1, few-shot)
  2. Entropy Baseline (forced-response entropy features + LogReg)
  3. Feature Vector (LLM-labeled chunk features + classifier sweep)
  4. BoW TF-IDF (TF-IDF + LogisticRegressionCV on CoT text)

Usage:
    python -m src2.runs.plot_min_maj_all_methods
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
PLOT_DIR = PROJECT_ROOT / "plots" / "min_maj_answer"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _compute_f1(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main():
    methods = {}

    # 1. LLM Monitor
    monitor_config = DATA_DIR / "loo_monitor" / "latest" / "run_config.json"
    monitor_folds = DATA_DIR / "loo_monitor" / "latest" / "loo_fold_metrics.csv"
    if monitor_config.exists():
        with open(monitor_config) as f:
            mc = json.load(f)
        pm = mc["pooled_metrics"]
        maj_f1 = _compute_f1(pm["majority_precision"], pm["majority_recall"])
        min_f1 = _compute_f1(pm["minority_precision"], pm["minority_recall"])
        macro_f1 = (maj_f1 + min_f1) / 2
        methods["LLM\nMonitor"] = {
            "macro_f1": macro_f1,
            "majority_f1": maj_f1,
            "minority_f1": min_f1,
            "accuracy": pm["accuracy"],
            "n": pm["n_total"],
        }
        # Load per-fold data if available
        if monitor_folds.exists():
            fdf = pd.read_csv(monitor_folds)
            # Compute per-fold macro F1 from precision/recall
            fold_f1s = []
            for _, row in fdf.iterrows():
                mf = _compute_f1(row["majority_precision"], row["majority_recall"])
                nf = _compute_f1(row["minority_precision"], row["minority_recall"])
                fold_f1s.append((mf + nf) / 2)
            methods["LLM\nMonitor"]["fold_f1s"] = fold_f1s
            methods["LLM\nMonitor"]["fold_labels"] = list(fdf["test_prompt_id"])
        print(f"LLM Monitor: macro_F1={macro_f1:.3f}")

    # 2. Entropy Baseline
    entropy_path = DATA_DIR / "entropy_baseline" / "baseline_results.json"
    if entropy_path.exists():
        with open(entropy_path) as f:
            eb = json.load(f)
        methods["Entropy\nBaseline"] = {
            "macro_f1": eb["pooled_macro_f1"],
            "majority_f1": eb["pooled_majority_f1"],
            "minority_f1": eb["pooled_minority_f1"],
            "accuracy": eb["pooled_accuracy"],
            "n": eb["n_total"],
            "fold_f1s": [fr["macro_f1"] for fr in eb["fold_results"]],
            "fold_labels": [fr["test_prompt_id"] for fr in eb["fold_results"]],
        }
        print(f"Entropy Baseline: macro_F1={eb['pooled_macro_f1']:.3f}")

    # 3. Feature Vector
    fv_path = DATA_DIR / "feature_vector" / "baseline_results.json"
    if fv_path.exists():
        with open(fv_path) as f:
            fv = json.load(f)
        # Use best config from sweep
        best = fv["top_20"][0] if fv.get("top_20") else None
        if best:
            methods["Feature\nVector"] = {
                "macro_f1": best["pooled_macro_f1"],
                "majority_f1": best["pooled_majority_f1"],
                "minority_f1": best["pooled_minority_f1"],
                "accuracy": best["pooled_accuracy"],
                "n": best["n_total"],
                "fold_f1s": [fr["macro_f1"] for fr in best["fold_results"]],
                "fold_labels": [fr["test_prompt_id"] for fr in best["fold_results"]],
            }
            print(f"Feature Vector: macro_F1={best['pooled_macro_f1']:.3f} "
                  f"({fv['best_config']})")

    # 4. BoW TF-IDF
    bow_config = DATA_DIR / "loo_bow_tfidf" / "latest" / "run_config.json"
    bow_folds = DATA_DIR / "loo_bow_tfidf" / "latest" / "loo_fold_metrics.csv"
    if bow_config.exists():
        with open(bow_config) as f:
            bc = json.load(f)
        pm = bc["pooled_metrics"]
        methods["BoW\nTF-IDF"] = {
            "macro_f1": pm["macro_f1"],
            "majority_f1": pm["majority_f1"],
            "minority_f1": pm["minority_f1"],
            "accuracy": pm["accuracy"],
            "n": pm["n_total"],
        }
        if bow_folds.exists():
            fdf = pd.read_csv(bow_folds)
            methods["BoW\nTF-IDF"]["fold_f1s"] = list(fdf["macro_f1"])
            methods["BoW\nTF-IDF"]["fold_labels"] = list(fdf["test_prompt_id"])
        print(f"BoW TF-IDF: macro_F1={pm['macro_f1']:.3f}")

    if not methods:
        print("No method results found!")
        return

    # ── Figure 1: Method comparison bar chart ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = list(methods.keys())
    n_methods = len(method_names)
    metrics = ["macro_f1", "majority_f1", "minority_f1", "accuracy"]
    metric_labels = ["Macro F1", "Majority F1", "Minority F1", "Accuracy"]
    x = np.arange(n_methods)
    width = 0.18
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [methods[m].get(metric, 0) for m in method_names]
        bars = ax.bar(x + i * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Majority/Minority Classification: Method Comparison\n(LOO Cross-Validation, Pooled)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    for i, m in enumerate(method_names):
        n = methods[m].get("n", 0)
        ax.text(i, -0.08, f"n={n}", ha="center", va="top", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "method_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {PLOT_DIR / 'method_comparison.png'}")

    # ── Figure 2: Per-fold macro F1 comparison ─────────────────────────
    methods_with_folds = {m: d for m, d in methods.items() if "fold_f1s" in d}
    if methods_with_folds:
        # Use fold labels from the first method that has them
        first = next(iter(methods_with_folds.values()))
        fold_labels = first["fold_labels"]
        n_folds = len(fold_labels)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_folds)
        n_m = len(methods_with_folds)
        width = 0.8 / n_m
        colors_fold = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63"]

        for i, (m_name, m_data) in enumerate(methods_with_folds.items()):
            f1s = m_data["fold_f1s"]
            # Pad if fewer folds
            while len(f1s) < n_folds:
                f1s.append(0)
            offset = (i - (n_m - 1) / 2) * width
            bars = ax.bar(x + offset, f1s[:n_folds], width,
                          label=m_name.replace("\n", " "),
                          color=colors_fold[i % len(colors_fold)], alpha=0.85)
            for bar, val in zip(bars, f1s[:n_folds]):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=6, fontweight="bold",
                    )

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
        ax.set_ylabel("Macro F1", fontsize=12)
        ax.set_title("Majority/Minority Classification: Per-Fold Macro F1\n(Leave-One-Prompt-Out)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontsize=9, rotation=30, ha="right")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(PLOT_DIR / "per_fold_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {PLOT_DIR / 'per_fold_comparison.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
