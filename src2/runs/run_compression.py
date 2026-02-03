"""
Run the CoT Compression pipeline across all rollouts for a question.

For each rollout:
  1. Prepare compression spec
  2. Run LLM sentence selection
  3. Evaluate compressed CoT vs baseline (50 resamples each)

Baseline is computed once and reused across all rollouts.

Usage:
    python -m src2.runs.run_compression
"""

import json
from pathlib import Path

from src2.methods import LlmMonitor
from src2.tasks import CompressedCotTask
from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SUBJECT_MODEL = "Qwen/Qwen3-32B"
MONITOR_MODEL = "openai/gpt-5.2"

COMPRESSION_FACTOR = 10
CHAR_LIMIT_MULTIPLIER = 1.5
COMPRESS_PCT = 0.5
REGION = "prefix"  # "prefix" or "middle"

QUESTION_ID = "starfish"
NUM_ROLLOUTS = 25
NUM_RESAMPLES = 50

task = CompressedCotTask(
    model=SUBJECT_MODEL,
    compression_factor=COMPRESSION_FACTOR,
    char_limit_multiplier=CHAR_LIMIT_MULTIPLIER,
    compress_pct=COMPRESS_PCT,
    region=REGION,
)

# only get baseline once
baseline_path = task.compression_dir / QUESTION_ID / "baseline_distribution.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline = json.load(f)
else:
    baseline = task.get_baseline_distribution(
        QUESTION_ID,
        rollout_idx=0,
        num_resamples=NUM_RESAMPLES,
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Saved baseline to {baseline_path}")

all_eval_results = []

for rollout_idx in range(NUM_ROLLOUTS):
    spec = task.run_data(question_id=QUESTION_ID, rollout_idx=rollout_idx)
    if spec is None:
        print(f"  Skipping rollout {rollout_idx} — run_data returned None")
        continue

    monitor_data = task.prepare_for_monitor()
    prompt = SentenceSelectionPrompt()
    monitor = LlmMonitor(prompt=prompt, model=MONITOR_MODEL)
    monitor.set_task(task)

    results = monitor.infer(monitor_data)

    eval_results = task.evaluate_monitor_results(
        results,
        spec,
        monitor.get_folder(),
        num_resamples=NUM_RESAMPLES,
        baseline=baseline,
    )
    all_eval_results.append(
        {
            "rollout_idx": rollout_idx,
            **eval_results,
        }
    )

    assert monitor._output is not None
    monitor._output.mark_success()

    js = eval_results["js_divergence"]
    agr = eval_results["agreement"]
    print(f"  JS divergence: {js:.4f}, Agreement: {agr:.1%}\n")

# save agg results
if all_eval_results:
    js_vals = [r["js_divergence"] for r in all_eval_results]
    agr_vals = [r["agreement"] for r in all_eval_results]

    summary_path = task.compression_dir / QUESTION_ID / "all_rollouts_eval.json"
    with open(summary_path, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"\n  Saved aggregate results to {summary_path}")
