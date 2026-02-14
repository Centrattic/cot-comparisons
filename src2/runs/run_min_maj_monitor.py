"""
Run the black-box monitor for majority/minority answer classification
using leave-one-out (LOO) cross-validation across all 7 prompts.

For each fold, one prompt is held out as test and the other 6 provide
few-shot examples. Results are aggregated as pooled metrics (all
predictions together) and mean-per-fold accuracy.

Usage:
    python -m src2.runs.run_min_maj_monitor
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src2.methods import LlmMonitor
from src2.tasks import MinMajAnswerTask
from src2.tasks.min_maj_answer.prompts import MinMajBlackBoxMonitorPrompt
from src2.tasks.min_maj_answer.task import ALL_PROMPT_IDS

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
LOO_DIR = DATA_DIR / "loo_monitor"

MONITOR_MODEL = "openai/gpt-4.1"
MAX_WORKERS = 30
REASONING_EFFORT = "medium"

N_EXAMPLES_PER_CLASS = 3  # per train prompt (6 prompts * 3 * 2 classes = 36 examples)
# ─────────────────────────────────────────────────────────────────────


def _create_run_dir() -> Path:
    """Create a timestamped run directory and update 'latest' symlink."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include model short name for easy identification
    model_short = MONITOR_MODEL.split("/")[-1]
    run_dir = LOO_DIR / f"{model_short}_{timestamp}"
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

# Initialize prompt template (shared across folds)
prompt = MinMajBlackBoxMonitorPrompt()

folds = MinMajAnswerTask.loo_folds()
all_fold_results = []
all_predictions = []
all_ground_truth = []
fold_metrics = []

print(f"\n{'=' * 60}")
print(f"  LOO Cross-Validation: {len(folds)} folds")
print(f"  Examples per class per train prompt: {N_EXAMPLES_PER_CLASS}")
print(f"{'=' * 60}")

for fold in folds:
    fold_idx = fold["fold_idx"]
    test_pid = fold["test_prompt_id"]
    train_pids = fold["train_prompt_ids"]

    print(f"\n--- Fold {fold_idx}: test={test_pid}, train={train_pids} ---")

    # Prepare monitor data for this fold
    monitor_data = task.get_monitor_data(
        test_prompt_ids=[test_pid],
        example_prompt_ids=train_pids,
        n_examples_per_class=N_EXAMPLES_PER_CLASS,
    )

    if not monitor_data:
        print(f"  No test data for {test_pid}, skipping")
        continue

    print(f"  Test rollouts: {len(monitor_data)}")

    # Initialize a fresh monitor per fold (separate output folder)
    monitor = LlmMonitor(
        prompt=prompt,
        model=MONITOR_MODEL,
        max_workers=MAX_WORKERS,
        temperature=0.3,
        max_tokens=1000,
        name=f"min_maj_monitor_fold{fold_idx}",
        reasoning_effort="medium",
    )
    monitor.set_task(task)

    # Run inference
    results = monitor.infer(monitor_data)
    monitor._output.mark_success()

    # Collect predictions
    preds = [r.get("monitor_prediction") for r in results]
    gt = [r.get("label") for r in results]
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
    all_fold_results.extend(results)

# ── Aggregate metrics ────────────────────────────────────────────────
pooled_metrics = task.evaluate(all_predictions, all_ground_truth)
fold_f1s = [fm["macro_f1"] for fm in fold_metrics]
mean_fold_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0

print(f"\n{'=' * 60}")
print(f"  LOO Aggregate Results ({len(fold_metrics)} folds)")
print(f"{'=' * 60}")
print(f"  Pooled macro F1:    {pooled_metrics['macro_f1']:.3f} "
      f"(n={pooled_metrics['n_total']})")
print(f"  Mean fold F1:       {mean_fold_f1:.3f}")
print(f"  Pooled accuracy:    {pooled_metrics['accuracy']:.3f}")
print(f"  Majority F1:        {pooled_metrics['majority_f1']:.3f}")
print(f"  Minority F1:        {pooled_metrics['minority_f1']:.3f}")

print(f"\n  Per-fold breakdown:")
for fm in fold_metrics:
    print(f"    Fold {fm['fold_idx']} ({fm['test_prompt_id']}): "
          f"F1={fm['macro_f1']:.3f}, acc={fm['accuracy']:.3f}, n={fm['n_total']}")

# ── Save results ─────────────────────────────────────────────────────
# Save per-fold metrics
fold_df = pd.DataFrame(fold_metrics)
fold_path = run_dir / "loo_fold_metrics.csv"
fold_df.to_csv(fold_path, index=False)

# Save all predictions with explanations
readable_rows = []
for r in all_fold_results:
    readable_rows.append({
        "prompt_id": r.get("prompt_id"),
        "rollout_idx": r.get("rollout_idx"),
        "answer": r.get("answer"),
        "label": r.get("label"),
        "prediction": r.get("monitor_prediction"),
        "correct": r.get("monitor_prediction") == r.get("label"),
        "monitor_explanation": r.get("monitor_response", ""),
    })

readable_df = pd.DataFrame(readable_rows)
readable_path = run_dir / "loo_results_with_explanations.csv"
readable_df.to_csv(readable_path, index=False)

# Save run config
run_config = {
    "monitor_model": MONITOR_MODEL,
    "reasoning_effort": REASONING_EFFORT,
    "n_examples_per_class": N_EXAMPLES_PER_CLASS,
    "n_folds": len(fold_metrics),
    "pooled_macro_f1": pooled_metrics["macro_f1"],
    "mean_fold_f1": mean_fold_f1,
    "pooled_accuracy": pooled_metrics["accuracy"],
    "pooled_metrics": pooled_metrics,
}
with open(run_dir / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print(f"\nResults saved to: {run_dir}")
print(f"  Fold metrics: {fold_path.name}")
print(f"  Full results: {readable_path.name}")
print(f"  Run config:   run_config.json")
