"""
BoW TF-IDF baseline for majority/minority answer classification
using leave-one-out (LOO) cross-validation across all 7 prompts.

For each fold, one prompt is held out as test and the other 6 are
used for training. Results are aggregated as pooled metrics and
mean-per-fold accuracy.

Usage:
    python -m src2.runs.run_min_maj_bow_tfidf
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src2.methods import BoWTfidf
from src2.tasks import MinMajAnswerTask
from src2.tasks.min_maj_answer.task import ALL_PROMPT_IDS

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
LOO_DIR = DATA_DIR / "loo_bow_tfidf"


def _create_run_dir() -> Path:
    """Create a timestamped run directory and update 'latest' symlink."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOO_DIR / f"bow_tfidf_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = LOO_DIR / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(run_dir.name, latest)
    return run_dir


# Initialize task with all prompts
task = MinMajAnswerTask(
    prompt_ids=ALL_PROMPT_IDS,
    model="qwen3-32b",
    data_dir=DATA_DIR,
)

# Generate/load data
task.run_data()
assert task.get_data(), "Data generation failed"

run_dir = _create_run_dir()
print(f"Run directory: {run_dir}")

folds = MinMajAnswerTask.loo_folds()
all_predictions = []
all_ground_truth = []
fold_metrics = []

print(f"\n{'=' * 60}")
print(f"  LOO Cross-Validation: {len(folds)} folds")
print(f"{'=' * 60}")

for fold in folds:
    fold_idx = fold["fold_idx"]
    test_pid = fold["test_prompt_id"]
    train_pids = fold["train_prompt_ids"]

    print(f"\n--- Fold {fold_idx}: test={test_pid}, train={train_pids} ---")

    # Get train/test split for this fold
    split = task.get_train_test_split(train_pids, [test_pid])
    train_df = split.train_df
    test_df = split.test_df

    if train_df.empty or test_df.empty:
        print(f"  No data for fold {fold_idx}, skipping")
        continue

    # Convert DataFrames to list-of-dicts
    train_entries = train_df[["cot_content", "label"]].to_dict("records")
    test_entries = test_df[["cot_content", "label"]].to_dict("records")

    print(f"  Train: {len(train_entries)}, Test: {len(test_entries)}")

    # Fresh method per fold
    method = BoWTfidf(
        text_key="cot_content",
        label_key="label",
        positive_label="minority",
        name=f"bow_tfidf_fold{fold_idx}",
    )
    method.set_task(task)
    method.train(train_entries)
    results = method.infer(test_entries)
    method._output.mark_success()

    # Map "no" -> "majority" in predictions
    preds = [
        r["prediction"] if r["prediction"] != "no" else "majority"
        for r in results
    ]
    gt = [r["label"] for r in results]

    metrics = task.evaluate(preds, gt)
    print(f"  Fold {fold_idx}: F1={metrics['macro_f1']:.3f}, "
          f"acc={metrics['accuracy']:.3f} (n={metrics['n_total']})")

    fold_metrics.append({
        "fold_idx": fold_idx,
        "test_prompt_id": test_pid,
        **metrics,
    })
    all_predictions.extend(preds)
    all_ground_truth.extend(gt)

# ── Aggregate metrics ────────────────────────────────────────────────
pooled_metrics = task.evaluate(all_predictions, all_ground_truth)
fold_f1s = [fm["macro_f1"] for fm in fold_metrics]
mean_fold_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0
fold_accs = [fm["accuracy"] for fm in fold_metrics]
mean_fold_acc = sum(fold_accs) / len(fold_accs) if fold_accs else 0.0

print(f"\n{'=' * 60}")
print(f"  LOO Aggregate Results ({len(fold_metrics)} folds)")
print(f"{'=' * 60}")
print(f"  Pooled macro F1:    {pooled_metrics['macro_f1']:.3f} "
      f"(n={pooled_metrics['n_total']})")
print(f"  Mean fold F1:       {mean_fold_f1:.3f}")
print(f"  Pooled accuracy:    {pooled_metrics['accuracy']:.3f}")
print(f"  Mean fold accuracy: {mean_fold_acc:.3f}")
print(f"  Majority F1:        {pooled_metrics['majority_f1']:.3f}")
print(f"  Minority F1:        {pooled_metrics['minority_f1']:.3f}")

print(f"\n  Per-fold breakdown:")
for fm in fold_metrics:
    print(f"    Fold {fm['fold_idx']} ({fm['test_prompt_id']}): "
          f"F1={fm['macro_f1']:.3f}, acc={fm['accuracy']:.3f}, n={fm['n_total']}")

# ── Save results ─────────────────────────────────────────────────────
fold_df = pd.DataFrame(fold_metrics)
fold_path = run_dir / "loo_fold_metrics.csv"
fold_df.to_csv(fold_path, index=False)

run_config = {
    "method": "bow_tfidf",
    "n_folds": len(fold_metrics),
    "pooled_macro_f1": pooled_metrics["macro_f1"],
    "mean_fold_f1": mean_fold_f1,
    "pooled_accuracy": pooled_metrics["accuracy"],
    "mean_fold_accuracy": mean_fold_acc,
    "pooled_metrics": pooled_metrics,
}
with open(run_dir / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print(f"\nResults saved to: {run_dir}")
print(f"  Fold metrics: {fold_path.name}")
print(f"  Run config:   run_config.json")
