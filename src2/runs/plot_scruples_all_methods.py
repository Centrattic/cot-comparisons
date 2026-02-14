#!/usr/bin/env python3
"""
Plot comparing all scruples sycophancy detection methods on the balanced test set.

Methods:
  1. Attention Probe (best config from hyperparameter sweep)
  2. Base Monitor v2 (LLM counterfactual monitor)
  3. High-Context Monitor (45 few-shot examples)
  4. Entropy Baseline (logistic regression on forced-response entropy features)
  5. BoW TF-IDF (TF-IDF + LogisticRegressionCV on CoT thinking text)

Also generates data scaling and training curves subplots.

Usage:
    python -m src2.runs.plot_scruples_all_methods
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"
PLOT_DIR = PROJECT_ROOT / "plots" / "scruples"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["suggest_wrong", "suggest_right"]


def _resolve_latest_run(base_dir: Path) -> Path:
    """Resolve the latest run directory, preferring 'latest' symlink,
    falling back to the most recent timestamped subdirectory."""
    latest = base_dir / "latest"
    if latest.exists():
        return latest
    timestamped = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    return latest


def compute_eval_monitor_metrics(monitor_name):
    """Compute F1/precision/recall from llm_monitor_eval_* directories.

    These CSVs have pre-computed columns:
      label (0/1), answer, monitor_prediction (A/B)
    Predicted label: monitor_prediction != answer -> sycophantic (1)
    """
    tp = fp = fn = tn = 0
    for variant in VARIANTS:
        base_dir = DATA_DIR / f"llm_monitor_eval_{monitor_name}_{variant}"
        if not base_dir.exists():
            print(f"  Warning: {base_dir} not found")
            continue
        run_dir = _resolve_latest_run(base_dir)
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["monitor_prediction"].notna()].copy()

        y_true = df["label"].values
        y_pred = (df["monitor_prediction"] != df["answer"]).astype(int).values

        tp += int(((y_pred == 1) & (y_true == 1)).sum())
        fp += int(((y_pred == 1) & (y_true == 0)).sum())
        fn += int(((y_pred == 0) & (y_true == 1)).sum())
        tn += int(((y_pred == 0) & (y_true == 0)).sum())

    total = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    acc = (tp + tn) / total if total > 0 else 0
    return {"f1": f1, "precision": p, "recall": r, "accuracy": acc, "n": total}


def main():
    # ── Collect all method results ─────────────────────────────────────
    methods = {}

    # 1. Attention Probe
    sweep_path = DATA_DIR / "sycophancy_probe" / "sweep_results.json"
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)
        best = sweep["best_test_metrics"]
        methods["Attention\nProbe"] = {
            "f1": best["per_class"]["sycophantic"]["f1"],
            "precision": best["per_class"]["sycophantic"]["precision"],
            "recall": best["per_class"]["sycophantic"]["recall"],
            "accuracy": best["accuracy"],
            "n": best["n_samples"],
        }
        print(f"Probe: F1={methods['Attention' + chr(10) + 'Probe']['f1']:.3f}")

    # 2. Base Monitor v2
    base_m = compute_eval_monitor_metrics("base_v2")
    if base_m["n"] > 0:
        methods["Base Monitor\n(v2)"] = base_m
        print(f"Base Monitor v2: F1={base_m['f1']:.3f} (n={base_m['n']})")

    # 3. High-Context Monitor
    hc_m = compute_eval_monitor_metrics("high_context")
    if hc_m["n"] > 0:
        methods["High-Context\nMonitor"] = hc_m
        print(f"High-Context Monitor: F1={hc_m['f1']:.3f} (n={hc_m['n']})")

    # 4. Entropy Baseline
    entropy_path = DATA_DIR / "entropy_baseline" / "baseline_results.json"
    if entropy_path.exists():
        with open(entropy_path) as f:
            ent = json.load(f)
        methods["Entropy\nBaseline"] = {
            "f1": ent["test_f1"],
            "precision": ent.get("test_precision", 0),
            "recall": ent.get("test_recall", 0),
            "accuracy": ent["test_accuracy"],
            "n": ent["n_test"],
        }
        print(f"Entropy Baseline: F1={ent['test_f1']:.3f}")
    else:
        print("Entropy baseline results not found (run --train phase first)")

    # 5. BoW TF-IDF
    bow_base = DATA_DIR / "bow_tfidf_scruples"
    if bow_base.exists():
        bow_run = _resolve_latest_run(bow_base)
        bow_results_path = bow_run / "results.json"
        if bow_results_path.exists():
            with open(bow_results_path) as f:
                bow = json.load(f)
            if "metrics" in bow:
                bm = bow["metrics"]
                methods["BoW\nTF-IDF"] = {
                    "f1": bm["f1"],
                    "precision": bm["precision"],
                    "recall": bm["recall"],
                    "accuracy": bm["accuracy"],
                    "n": len(bow.get("predictions", [])),
                }
                print(f"BoW TF-IDF: F1={bm['f1']:.3f}")

    if not methods:
        print("No method results found!")
        return

    # ── Figure 1: Method comparison bar chart ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = list(methods.keys())
    n_methods = len(method_names)
    metric_keys = ["f1", "precision", "recall", "accuracy"]
    metric_labels = ["F1", "Precision", "Recall", "Accuracy"]
    x = np.arange(n_methods)
    width = 0.18
    colors = ["#4E79A7", "#59A14F", "#E15759", "#F28E2B"]

    for i, (metric, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [methods[m].get(metric, 0) for m in method_names]
        bars = ax.bar(x + i * width - 1.5 * width, vals, width,
                      label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sycophancy Detection: Method Comparison\n(Balanced Test Set)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "method_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {PLOT_DIR / 'method_comparison.png'}")

    # ── Figure 2: Data scaling ─────────────────────────────────────────
    scaling_path = DATA_DIR / "data_scaling" / "scruples_scaling.json"
    if scaling_path.exists():
        with open(scaling_path) as f:
            sc = json.load(f)

        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = np.array(sc["sizes"])
        mean_f1 = np.array(sc["mean_f1"])
        std_f1 = np.array(sc["std_f1"])

        ax.plot(sizes, mean_f1, "b-o", markersize=5, label="Test F1")
        ax.fill_between(sizes, mean_f1 - std_f1, mean_f1 + std_f1, color="blue", alpha=0.15)

        # Add train metrics if available
        if "mean_train_f1" in sc:
            tr_f1 = np.array(sc["mean_train_f1"])
            tr_std = np.array(sc["std_train_f1"])
            ax.plot(sizes, tr_f1, "r-o", markersize=5, label="Train F1")
            ax.fill_between(sizes, tr_f1 - tr_std, tr_f1 + tr_std, color="red", alpha=0.15)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("F1 (Sycophantic Class)", fontsize=12)
        ax.set_title("Sycophancy Probe: Data Scaling", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "data_scaling.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {PLOT_DIR / 'data_scaling.png'}")

    # ── Figure 3: Hyperparameter sweep ─────────────────────────────────
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)

        fig, ax = plt.subplots(figsize=(9, 5))
        configs = sweep["sweep_results"]
        configs_sorted = sorted(configs, key=lambda c: c["test_f1_syc"], reverse=True)

        labels = [f"wd={c['weight_decay']:.0e}\ndo={c['dropout']}" for c in configs_sorted]
        test_f1s = [c["test_f1_syc"] for c in configs_sorted]
        val_f1s = [c["best_val_f1"] for c in configs_sorted]

        x = np.arange(len(configs_sorted))
        ax.bar(x - 0.15, test_f1s, 0.3, label="Test F1", color="#3A86FF", alpha=0.85)
        ax.bar(x + 0.15, val_f1s, 0.3, label="Val F1", color="#FFD166", alpha=0.85)

        for i, (tf, vf) in enumerate(zip(test_f1s, val_f1s)):
            ax.text(i - 0.15, tf + 0.01, f"{tf:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + 0.15, vf + 0.01, f"{vf:.3f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_ylabel("F1", fontsize=12)
        ax.set_title("Attention Probe: Hyperparameter Sweep", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "hyperparam_sweep.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {PLOT_DIR / 'hyperparam_sweep.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
