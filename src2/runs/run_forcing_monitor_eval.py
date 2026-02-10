#!/usr/bin/env python3
"""
Evaluate forcing LLM monitor as a black-box baseline for the entropy probe.

Runs ForcingMonitorPrompt (text-only, no activations) on the exact same eval
split as the entropy probe, derives entropy from predicted distributions, and
computes comparable metrics (MSE, R², Pearson r, per-question breakdown).

Usage:
    python -m src2.runs.run_forcing_monitor_eval
"""

import json
from pathlib import Path

import numpy as np

from src2.data_slice import DataSlice
from src2.methods.llm_monitor import LlmMonitor
from src2.tasks.forced_response.prompts import ForcingMonitorPrompt
from src2.tasks.forced_response.task import ForcingTask
from src2.runs.run_entropy_probe import (
    build_question_splits,
    sample_sentence_indices,
    dist_dict_to_entropy,
    compute_metrics,
    print_results,
    SUBJECT_MODEL,
    DATA_DIR,
    MAX_SENTENCES_PER_QUESTION_EVAL,
    SEED,
)

# ── Configuration ─────────────────────────────────────────────────────
MONITOR_MODEL = "openai/gpt-5.2"
MAX_WORKERS = 50


# ── Main ──────────────────────────────────────────────────────────────


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # ── 1. Build question splits (identical to entropy probe) ─────────
    print("Building question splits (identical to entropy probe)...")
    splits = build_question_splits(forcing.forcing_dir)
    eval_ids = splits["eval_ids"]
    print(f"  Eval: {len(eval_ids)} questions")
    print(f"  Eval IDs: {eval_ids}")

    # ── 2. Sample sentence indices (identical to entropy probe) ───────
    print("\nSampling sentence indices...")
    eval_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, eval_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    eval_sents = sum(len(v) for v in eval_sentence_map.values())
    print(f"  Eval: {eval_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_EVAL}/question)")

    # ── 3. Load text data for eval split (no activations) ─────────────
    print("\nLoading eval data (text-only)...")
    eval_ds = DataSlice(
        ids=set(eval_ids),
        sentence_indices=None,  # filter per-question below
    )

    # Load question objects for question text + choices
    question_cache = {}
    for qid in eval_ids:
        loaded = forcing.load_question_and_cot(qid, rollout_idx=0)
        if loaded is not None:
            question_cache[qid] = loaded[0]  # Question object
        else:
            print(f"  WARNING: could not load question for {qid}")

    # Iterate sentence-level summaries matching eval split
    eval_rows = []
    ground_truth_entropies = []
    row_metadata = []  # parallel arrays for question_id, sentence_idx

    for qid in eval_ids:
        allowed_sentences = eval_sentence_map.get(qid, set())
        if not allowed_sentences:
            continue

        question = question_cache.get(qid)
        if question is None:
            continue

        # Find all sentence summaries for this question
        summaries = sorted(
            forcing.forcing_dir.glob(f"{qid}/rollout_*/*/sentence_*/summary.json")
        )

        for sp in summaries:
            with open(sp) as f:
                summary = json.load(f)

            sentence_idx = summary.get("sentence_idx", -1)
            if sentence_idx not in allowed_sentences:
                continue

            # Ground truth distribution and entropy
            answer_dist = summary.get("answer_distribution")
            if answer_dist is None:
                answer_counts = summary.get("answer_counts", {})
                total = sum(answer_counts.values())
                if total == 0:
                    continue
                answer_dist = {k: v / total for k, v in answer_counts.items()}

            if not answer_dist or all(v == 0 for v in answer_dist.values()):
                continue

            gt_entropy = dist_dict_to_entropy(answer_dist)
            if gt_entropy is None:
                continue

            partial_cot = summary.get("partial_cot", "")
            question_type = summary.get("question_type", "multiple_choice")

            row = {
                "question_id": qid,
                "sentence_idx": sentence_idx,
                "question": question.question,
                "partial_cot": partial_cot,
                "question_type": question_type,
            }

            if hasattr(question, "choices"):
                row["choices"] = question.choices

            eval_rows.append(row)
            ground_truth_entropies.append(gt_entropy)
            row_metadata.append({"question_id": qid, "sentence_idx": sentence_idx})

    print(f"  Eval rows: {len(eval_rows)}")
    print(f"  Questions with data: {len(set(m['question_id'] for m in row_metadata))}")

    if not eval_rows:
        print("No eval data found. Exiting.")
        return

    # ── 4. Run monitor ────────────────────────────────────────────────
    print(f"\nRunning ForcingMonitorPrompt with {MONITOR_MODEL}...")
    monitor = LlmMonitor(
        prompt=ForcingMonitorPrompt(),
        model=MONITOR_MODEL,
        max_workers=MAX_WORKERS,
        name="llm_monitor_forcing_monitor_eval",
    )
    monitor.set_task(forcing)
    results = monitor.infer(eval_rows)
    monitor._output.mark_success()

    # ── 5. Parse predictions and compute entropy ──────────────────────
    print("\nParsing monitor predictions...")
    y_true = []
    y_pred = []
    question_ids = []
    n_valid = 0
    n_failed = 0

    prompt_obj = ForcingMonitorPrompt()

    for i, res in enumerate(results):
        pred_dist = res.get("monitor_prediction")
        if pred_dist is None:
            n_failed += 1
            continue

        pred_entropy = dist_dict_to_entropy(pred_dist)
        if pred_entropy is None:
            n_failed += 1
            continue

        y_true.append(ground_truth_entropies[i])
        y_pred.append(pred_entropy)
        question_ids.append(row_metadata[i]["question_id"])
        n_valid += 1

    print(f"  Valid predictions: {n_valid}")
    print(f"  Failed/unparseable: {n_failed}")

    if n_valid == 0:
        print("No valid predictions. Exiting.")
        return

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    # ── 6. Compute and print metrics ──────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, question_ids)
    print_results(metrics, label="Forcing Monitor Eval (black-box baseline)")

    # ── 7. Save results ───────────────────────────────────────────────
    output_dir = DATA_DIR / "monitor_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output = {
        "metrics": metrics,
        "config": {
            "subject_model": SUBJECT_MODEL,
            "monitor_model": MONITOR_MODEL,
            "max_workers": MAX_WORKERS,
            "eval_question_ids": eval_ids,
            "n_eval_rows": len(eval_rows),
            "n_valid_predictions": n_valid,
            "n_failed_predictions": n_failed,
            "max_sentences_per_question": MAX_SENTENCES_PER_QUESTION_EVAL,
            "seed": SEED,
        },
    }
    with open(output_dir / "forcing_monitor_results.json", "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    print(f"\nResults saved to {output_dir / 'forcing_monitor_results.json'}")


if __name__ == "__main__":
    main()
