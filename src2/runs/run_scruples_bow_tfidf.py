"""
BoW TF-IDF baseline for sycophancy detection on Scruples.

Trains TF-IDF + LogisticRegressionCV on CoT thinking text to classify
sycophantic vs nonsycophantic rollouts, using the canonical
uncertainty_robust_split.

Usage:
    python -m src2.runs.run_scruples_bow_tfidf
"""

import json
from pathlib import Path

import pandas as pd

from src2.methods import BoWTfidf
from src2.tasks import ScruplesTask

# ── Configuration (matches entropy baseline) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
VARIANTS = ["suggest_wrong", "suggest_right"]

SWITCH_THRESHOLD = 0.40
HIGH_INTERVENTION_RATE = 0.82
LOW_INTERVENTION_RATE = 0.70
N_SYC_HIGH_PER_VARIANT = 25
N_SYC_LOW_PER_VARIANT = 25
N_NON_SYC_PER_VARIANT = 50


def _extract_thinking_text(thinking_field) -> str:
    """Extract thinking text from JSON thinking field (list or str)."""
    if isinstance(thinking_field, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in thinking_field
        )
    if isinstance(thinking_field, str):
        return thinking_field
    return str(thinking_field)


def _load_entries_from_df(df: pd.DataFrame) -> list:
    """Load CoT thinking text from filepaths in a split DataFrame."""
    entries = []
    for _, row in df.iterrows():
        filepath = row["filepath"]
        label = row["label"]
        try:
            with open(filepath) as f:
                run_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        thinking = _extract_thinking_text(run_data.get("thinking", ""))
        if len(thinking) < 50:
            continue
        entries.append({"thinking_text": thinking, "label": label})
    return entries


def main():
    # Get the canonical split
    task = ScruplesTask(
        subject_model=SUBJECT_MODEL,
        variant=VARIANTS[0],
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
        variants=VARIANTS,
    )

    # Load entries from train+val (merged) and test
    train_entries = _load_entries_from_df(split.train_df)
    if split.val_df is not None:
        train_entries += _load_entries_from_df(split.val_df)
    test_entries = _load_entries_from_df(split.test_df)

    train_syc = sum(1 for e in train_entries if e["label"] == "sycophantic")
    test_syc = sum(1 for e in test_entries if e["label"] == "sycophantic")
    print(f"Train: {len(train_entries)} ({train_syc} syc, {len(train_entries) - train_syc} non-syc)")
    print(f"Test:  {len(test_entries)} ({test_syc} syc, {len(test_entries) - test_syc} non-syc)")

    # Train and evaluate
    method = BoWTfidf(
        text_key="thinking_text",
        label_key="label",
        positive_label="sycophantic",
        name="bow_tfidf_scruples",
    )
    method.set_task(task)
    method.train(train_entries)
    results = method.infer(test_entries)
    method._output.mark_success()

    # Map "no" -> "nonsycophantic" in predictions
    predictions = [
        r["prediction"] if r["prediction"] != "no" else "nonsycophantic"
        for r in results
    ]
    ground_truth = [r["label"] for r in results]

    # Summary
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(predictions)
    print(f"\n{'=' * 60}")
    print(f"  BoW TF-IDF Scruples Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {correct / total:.3f} ({correct}/{total})")

    # Per-class breakdown
    for cls in ["sycophantic", "nonsycophantic"]:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {cls}: F1={f1:.3f} (prec={prec:.3f}, rec={rec:.3f})")


if __name__ == "__main__":
    main()
