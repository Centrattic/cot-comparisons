"""
Run the black-box monitor for majority/minority answer classification.

Few-shot examples are drawn from 5 reference prompts:
  bagel, gpqa_nmr_compound, gpqa_benzene_naming, harder_well, bookworm

Test prompts (all rollouts classified):
  gpqa_diels_alder, gpqa_optical_activity

Usage:
    python -m src2.runs.run_min_maj_monitor
"""

from pathlib import Path

import pandas as pd

from src2.methods import LlmMonitor
from src2.tasks import MinMajAnswerTask
from src2.tasks.min_maj_answer.prompts import MinMajBlackBoxMonitorPrompt

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"

MONITOR_MODEL = "openai/gpt-4.1"
MAX_WORKERS = 20

# Reference prompts for few-shot examples (spread across these prompts)
EXAMPLE_PROMPT_IDS = [
    "bagel",
    "gpqa_nmr_compound",
    "gpqa_benzene_naming",
    "harder_well",
    "bookworm",
]

# Test prompts — classify every rollout
TEST_PROMPT_IDS = [
    "gpqa_diels_alder",
    "gpqa_optical_activity",
]

N_EXAMPLES_PER_CLASS = 3  # 3 majority + 3 minority total (spread across example prompts)
# ─────────────────────────────────────────────────────────────────────

# Initialize task with all prompts (examples + test)
all_prompt_ids = EXAMPLE_PROMPT_IDS + TEST_PROMPT_IDS
task = MinMajAnswerTask(
    prompt_ids=all_prompt_ids,
    model="qwen3-32b",
    data_dir=DATA_DIR,
)

# Generate/load data
task.run_data()
assert task.get_data(), "Data generation failed"

# Prepare monitor data (few-shot examples from reference prompts, test on test prompts)
monitor_data = task.get_monitor_data(
    test_prompt_ids=TEST_PROMPT_IDS,
    example_prompt_ids=EXAMPLE_PROMPT_IDS,
    n_examples_per_class=N_EXAMPLES_PER_CLASS,
)

print(f"\nTest set: {len(monitor_data)} rollouts to classify")
print(f"Few-shot examples: {N_EXAMPLES_PER_CLASS} majority + {N_EXAMPLES_PER_CLASS} minority "
      f"= {N_EXAMPLES_PER_CLASS * 2} total (spread across {len(EXAMPLE_PROMPT_IDS)} prompts)")

# Initialize monitor
prompt = MinMajBlackBoxMonitorPrompt(cot_max_chars=2000)
monitor = LlmMonitor(
    prompt=prompt,
    model=MONITOR_MODEL,
    max_workers=MAX_WORKERS,
    temperature=0.3,
    max_tokens=1000,
)
monitor.set_task(task)

# Run inference
results = monitor.infer(monitor_data)
monitor._output.mark_success()

# Evaluate
predictions = [r.get("monitor_prediction") for r in results]
ground_truth = [r.get("label") for r in results]
metrics = task.evaluate(predictions, ground_truth)

print("\n=== Results ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")

# Per-prompt breakdown
for pid in TEST_PROMPT_IDS:
    pid_results = [r for r in results if r.get("prompt_id") == pid]
    pid_preds = [r.get("monitor_prediction") for r in pid_results]
    pid_gt = [r.get("label") for r in pid_results]
    pid_metrics = task.evaluate(pid_preds, pid_gt)
    print(f"\n  {pid} (n={len(pid_results)}):")
    for k, v in pid_metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v}")

# Save readable results CSV with monitor explanations
readable_rows = []
for r in results:
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
readable_path = monitor.get_folder() / "results_with_explanations.csv"
readable_df.to_csv(readable_path, index=False)

print(f"\nResults saved to: {monitor.get_folder()}")
print(f"Monitor explanations: {readable_path}")
