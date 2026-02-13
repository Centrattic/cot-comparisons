#!/usr/bin/env python3
"""
Plot entropy vs. % CoT completion for sycophantic vs nonsycophantic rollouts.

Loads forcing checkpoint data and the uncertainty-robust test split, matches
rollouts by (anecdote_id, variant, run_idx), then plots mean entropy with 95%
CI bands across percent-complete bins.

Usage:
    python -m src2.runs.plot_entropy_trajectory
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src2.tasks import ScruplesTask

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"
OUTPUT_DIR = DATA_DIR / "entropy_baseline"
CHECKPOINT_PATH = OUTPUT_DIR / "forcing_checkpoint.json"
OUTPUT_PNG = OUTPUT_DIR / "entropy_trajectory_test.png"

# Must match the split config used in run_scruples_entropy_baseline.py
SUBJECT_MODEL = "Qwen/Qwen3-32B"
SWITCH_THRESHOLD = 0.40
HIGH_INTERVENTION_RATE = 0.82
LOW_INTERVENTION_RATE = 0.70
N_SYC_HIGH_PER_VARIANT = 25
N_SYC_LOW_PER_VARIANT = 25
N_NON_SYC_PER_VARIANT = 50
TEST_SPLIT = 0.2
SEED = 42

BIN_WIDTH = 5  # percent


def main():
    # 1. Load forcing checkpoint
    print(f"Loading forcing data from {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)
    rollouts = checkpoint["rollouts"]
    print(f"  {len(rollouts)} total rollouts")

    # 2. Get test split
    task = ScruplesTask(
        subject_model=SUBJECT_MODEL,
        variant="suggest_wrong",
        data_dir=DATA_DIR,
    )
    split = task.get_uncertainty_robust_split(
        switch_threshold=SWITCH_THRESHOLD,
        non_syc_max_switch=0.10,
        high_intervention_rate=HIGH_INTERVENTION_RATE,
        low_intervention_rate=LOW_INTERVENTION_RATE,
        n_syc_high_per_variant=N_SYC_HIGH_PER_VARIANT,
        n_syc_low_per_variant=N_SYC_LOW_PER_VARIANT,
        n_non_syc_per_variant=N_NON_SYC_PER_VARIANT,
        test_split=TEST_SPLIT,
        seed=SEED,
    )
    test_df = split.test_df
    print(f"  Test set: {len(test_df)} rows")

    # 3. Build lookup: (anecdote_id, variant, run_idx) -> label
    test_keys = {}
    for _, row in test_df.iterrows():
        key = (row["anecdote_id"], row["variant"], int(row["run_idx"]))
        test_keys[key] = row["label"]

    # 4. Flatten rollout results into rows with percent_complete
    rows = []
    matched = 0
    for r in rollouts:
        key = (r["anecdote_id"], r["variant"], int(r["run_idx"]))
        label = test_keys.get(key)
        if label is None:
            continue
        matched += 1
        for pt in r["results"]:
            pct = (1.0 - pt["fraction_remaining"]) * 100.0
            rows.append({
                "percent_complete": pct,
                "entropy": pt["entropy"],
                "label": label,
            })

    print(f"  Matched {matched} rollouts to test set")
    df = pd.DataFrame(rows)

    # 5. Bin by percent_complete
    bin_edges = np.arange(0, 100 + BIN_WIDTH, BIN_WIDTH)
    df["bin"] = pd.cut(
        df["percent_complete"],
        bins=bin_edges,
        include_lowest=True,
        labels=False,
    )
    # Bin center for plotting
    df["bin_center"] = df["bin"] * BIN_WIDTH + BIN_WIDTH / 2.0

    # 6. Compute percentile bands per bin per label
    #    Layered bands: denser regions get more opacity
    BANDS = [
        (0.0, 1.0, 0.08),    # min–max, lightest
        (0.10, 0.90, 0.12),   # 10–90
        (0.25, 0.75, 0.18),   # 25–75, darkest band
    ]

    def _agg(group):
        ent = group["entropy"]
        result = {"median": ent.median(), "n": len(ent)}
        for lo, hi, _ in BANDS:
            result[f"p{int(lo*100)}"] = ent.quantile(lo)
            result[f"p{int(hi*100)}"] = ent.quantile(hi)
        return pd.Series(result)

    agg = df.groupby(["label", "bin_center"]).apply(_agg, include_groups=False).reset_index()

    # 7. Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    label_style = {
        "sycophantic": {"color": "#d62728", "label": "Sycophantic"},
        "nonsycophantic": {"color": "#1f77b4", "label": "Non-sycophantic"},
    }

    for label, style in label_style.items():
        sub = agg[agg["label"] == label].sort_values("bin_center")
        if sub.empty:
            continue
        x = sub["bin_center"].values
        # Draw bands from widest (lightest) to narrowest (darkest)
        for lo, hi, alpha in BANDS:
            lo_vals = sub[f"p{int(lo*100)}"].values
            hi_vals = sub[f"p{int(hi*100)}"].values
            ax.fill_between(x, lo_vals, hi_vals, color=style["color"], alpha=alpha)
        # Median line on top
        ax.plot(x, sub["median"].values, color=style["color"],
                label=style["label"], linewidth=2)

    ax.set_xlabel("% of CoT completed", fontsize=12)
    ax.set_ylabel("Shannon entropy (bits)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved plot to {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
