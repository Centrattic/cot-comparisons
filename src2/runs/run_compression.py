"""
Run the CoT Compression pipeline.

Three compression methods:
  1. LLM sentence selection (pick important sentence indices)
  2. LLM summary compression (rewrite region section)
  3. Thought anchors (adaptive resampling importance scoring)

Supports configurable compression region (prefix or middle) and percentage.

Usage:
    python -m src2.runs.run_compression
"""

from pathlib import Path

from src2.methods import LlmMonitor, ThoughtAnchors
from src2.tasks import CompressedCotTask
from src2.tasks.compressed_cot.prompts import (
    SentenceSelectionPrompt,
    SummaryCompressionPrompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SUBJECT_MODEL = "Qwen/Qwen3-32B"
MONITOR_MODEL = "openai/gpt-5.2"

COMPRESSION_FACTOR = 10
CHAR_LIMIT_MULTIPLIER = 1.5
COMPRESS_PCT = 0.5
REGION = "prefix"  # "prefix" or "middle"

QUESTION_ID = "starfish"
ROLLOUT_IDX = 0
NUM_EVAL_RESAMPLES = 50

GENERATE_DATA = True
RUN_SENTENCE_SELECTION = True
RUN_SUMMARY_COMPRESSION = True
RUN_THOUGHT_ANCHORS = True
EVALUATE = True

task = CompressedCotTask(
    model=SUBJECT_MODEL,
    compression_factor=COMPRESSION_FACTOR,
    char_limit_multiplier=CHAR_LIMIT_MULTIPLIER,
    compress_pct=COMPRESS_PCT,
    region=REGION,
)

spec = task.run_data(question_id=QUESTION_ID, rollout_idx=ROLLOUT_IDX)

assert spec is not None
assert task.get_data()

monitor_data = task.prepare_for_monitor()

compressed_cots = {}

prompt = SentenceSelectionPrompt()
monitor = LlmMonitor(prompt=prompt, model=MONITOR_MODEL)
monitor.set_task(task)
results = monitor.infer(monitor_data)

monitor._output.mark_success()

# # Extract compressed CoTs from sentence selection results
# for r in results:
#     indices = r.get("monitor_prediction")
#     if indices and isinstance(indices, list):
#         sentences = r.get("sentences", [])
#         region_start = r.get("region_start_idx", 0)
#         region_end = r.get("region_end_idx", len(sentences))
#         region = sentences[region_start:region_end]
#         # Indices from the prompt are absolute; convert to relative
#         relative = [
#             i - region_start for i in indices if region_start <= i < region_end
#         ]
#         kept = " ".join(
#             region[i] for i in sorted(relative) if 0 <= i < len(region)
#         )
#         pre_region = " ".join(sentences[:region_start])
#         post_region = " ".join(sentences[region_end:])
#         parts = [p for p in [pre_region, kept, post_region] if p]
#         compressed_cots["sentence_selection"] = " ".join(parts)

# # Step 2b: LLM summary compression
# if RUN_SUMMARY_COMPRESSION:
# prompt = SummaryCompressionPrompt()
# monitor = LlmMonitor(prompt=prompt, model=MONITOR_MODEL)
# monitor.set_task(task)
# folder = monitor.get_folder()
# results = monitor.infer(monitor_data)
# monitor._output.mark_success()
# print(f"Summary compression results saved to: {folder}")

# for r in results:
#     summary_text = r.get("monitor_prediction")
#     if summary_text:
#         sentences = r.get("sentences", [])
#         region_start = r.get("region_start_idx", 0)
#         region_end = r.get("region_end_idx", len(sentences))
#         pre_region = " ".join(sentences[:region_start])
#         post_region = " ".join(sentences[region_end:])
#         parts = [p for p in [pre_region, summary_text, post_region] if p]
#         compressed_cots["summary_compression"] = " ".join(parts)

# # Step 2c: Thought anchors (adaptive resampling)
# if RUN_THOUGHT_ANCHORS:
# anchors = ThoughtAnchors(model=SUBJECT_MODEL)
# anchors.set_task(task)
# folder = anchors.get_folder()
# anchor_data = task.prepare_for_thought_anchors()
# results = anchors.infer(anchor_data)
# anchors._output.mark_success()
# print(f"Thought anchors results saved to: {folder}")

# for r in results:
#     if r.get("compressed_cot"):
#         compressed_cots["thought_anchors"] = r["compressed_cot"]

# # Step 3: Evaluate each compressed CoT
# if EVALUATE and compressed_cots:
# print("\n=== Evaluation ===")
# print(f"Getting baseline distribution ({NUM_EVAL_RESAMPLES} resamples)...")
# baseline = task.get_baseline_distribution(
#     QUESTION_ID,
#     ROLLOUT_IDX,
#     num_resamples=NUM_EVAL_RESAMPLES,
# )
# print(f"Baseline: {baseline['distribution']}")

# eval_results = {}
# for method_name, compressed_cot in compressed_cots.items():
#     print(f"\nEvaluating {method_name}...")
#     result = task.evaluate_compression(
#         QUESTION_ID,
#         ROLLOUT_IDX,
#         compressed_cot,
#         num_resamples=NUM_EVAL_RESAMPLES,
#     )
#     eval_results[method_name] = result
#     print(f"  Distribution: {result['distribution']}")

#     metrics = task.evaluate(
#         [result["distribution"]],
#         [baseline["distribution"]],
#     )
#     print(f"  JS divergence from baseline: {metrics['js_divergence']:.4f}")
#     print(f"  Top answer agreement: {metrics['agreement']:.1%}")

# print("\n=== Summary ===")
# for method_name, result in eval_results.items():
#     print(
#         f"  {method_name}: top={result['most_common']}, "
#         f"agreement={result['agreement_rate']:.1%}"
#     )
