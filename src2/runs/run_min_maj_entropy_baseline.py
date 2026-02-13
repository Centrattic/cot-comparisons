#!/usr/bin/env python3
"""
Entropy baseline for majority/minority answer classification.

Tests whether forced-answer entropy alone can distinguish majority
from minority rollouts. Two-phase script:

  Phase 1 (GENERATE): Run forced-response on rollouts to get entropy
    trajectories at each sentence boundary. Requires Tinker.

  Phase 2 (TRAIN): Load the saved entropy data, extract features
    (mean, final, std, etc.), and train logistic regression to
    classify majority vs minority using LOO cross-validation
    across all 7 prompts. Reports per-fold and aggregate metrics.

All prompts use N/M answer format from verification_rollouts.

Usage:
    # Phase 1: generate forcing data (requires Tinker)
    python -m src2.runs.run_min_maj_entropy_baseline --generate

    # Phase 2: train baseline with LOO-CV (no Tinker needed)
    python -m src2.runs.run_min_maj_entropy_baseline --train

    # Both
    python -m src2.runs.run_min_maj_entropy_baseline --generate --train
"""

import argparse
import json
import math
import os
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from src2.tasks.min_maj_answer.task import ALL_PROMPT_IDS, MinMajAnswerTask

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
OUTPUT_DIR = DATA_DIR / "entropy_baseline"
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"
QUESTIONS_FILE = PROJECT_ROOT / "src2" / "utils" / "questions.json"

SUBJECT_MODEL = "Qwen/Qwen3-32B"

# Entropy baseline uses _nm suffix variants of the prompt IDs
NM_SUFFIX = "_nm"
ALL_PROMPT_IDS_NM = [pid + NM_SUFFIX for pid in ALL_PROMPT_IDS]

TOPK = 20
SENTENCE_STRIDE = 1

SEED = 42


def _create_run_dir(phase: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{phase}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = OUTPUT_DIR / f"latest_{phase}"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(run_dir.name, latest)
    return run_dir


def _shannon_entropy(probs: Dict[str, float]) -> float:
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ── Verification rollout loading ──────────────────────────────────────


def _get_latest_verification_dir(question_id: str) -> Optional[Path]:
    """Find the latest timestamped run directory for a verification question."""
    qdir = VERIFICATION_DIR / question_id
    if not qdir.exists():
        return None
    timestamped = sorted(
        [d for d in qdir.iterdir()
         if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"],
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    if (qdir / "summary.json").exists():
        return qdir
    return None


def _load_verification_rollouts(question_id: str) -> pd.DataFrame:
    """
    Load rollouts from verification_rollouts directory.

    Computes majority/minority labels from answer distribution.
    Returns a DataFrame with columns:
      prompt_id, rollout_idx, cot_content, answer, label
    """
    run_dir = _get_latest_verification_dir(question_id)
    if run_dir is None:
        raise FileNotFoundError(
            f"No verification rollouts found for {question_id} in {VERIFICATION_DIR}. "
            f"Run run_min_maj_verification.py first."
        )

    rollouts_dir = run_dir / "rollouts"
    raw_rollouts = []
    for f in sorted(rollouts_dir.iterdir()):
        if f.suffix == ".json":
            with open(f) as fh:
                raw_rollouts.append(json.load(fh))

    # Compute majority/minority labels
    answers = [r["answer"] for r in raw_rollouts if r.get("answer")]
    counts = Counter(answers)
    if not counts:
        return pd.DataFrame()
    majority_answer = counts.most_common(1)[0][0]

    rows = []
    for r in raw_rollouts:
        ans = r.get("answer", "")
        if not ans:
            continue
        rows.append({
            "prompt_id": question_id,
            "rollout_idx": r["rollout_idx"],
            "cot_content": r.get("thinking", ""),
            "answer": ans,
            "label": "majority" if ans == majority_answer else "minority",
        })

    df = pd.DataFrame(rows)
    n_maj = (df["label"] == "majority").sum()
    n_min = (df["label"] == "minority").sum()
    print(f"  {question_id}: {len(df)} rollouts, "
          f"{n_maj} majority ({majority_answer}), {n_min} minority")
    return df


def _build_user_msg(question_id: str) -> str:
    """
    Build the user message for a question, matching what was used
    during verification rollout generation.
    """
    from src2.utils.questions import load_custom_questions
    from src2.utils.verification import _format_user_message

    questions = load_custom_questions(QUESTIONS_FILE)
    for q in questions:
        if q.id == question_id:
            return _format_user_message(q)
    raise ValueError(f"Question {question_id} not found in {QUESTIONS_FILE}")


# ── Phase 1: Generate forcing data ───────────────────────────────────


def generate_forcing_data():
    """Run forced-response on all rollouts from train + test."""
    import contextlib
    import io
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tinker import ServiceClient, types
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from src2.tasks.forced_response.prompts import get_cumulative_cot_segments
    from src2.utils.chat_template import build_thinking_prompt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Setup Tinker ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=SUBJECT_MODEL)

    # ── Load all rollouts from verification_rollouts ──────────────────
    all_prompt_ids = ALL_PROMPT_IDS_NM
    dfs = []
    user_msgs = {}
    for qid in all_prompt_ids:
        df = _load_verification_rollouts(qid)
        dfs.append(df)
        user_msgs[qid] = _build_user_msg(qid)

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rollouts to force: {len(all_df)} ({all_df['label'].value_counts().to_dict()})")

    # ── Resolve A/B token IDs ─────────────────────────────────────────
    # All prompts use N/M answer format
    choices = ["N", "M"]
    answer_token_ids = {}
    for ans in choices:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = tokenizer.encode(ans, add_special_tokens=False)
        answer_token_ids[ans] = ids[-1]
    print(f"Answer token IDs: {answer_token_ids}")

    assert answer_token_ids["N"] != answer_token_ids["M"], \
        "N and M map to the same token ID!"

    def _get_choice_probs(prompt_str: str) -> Dict[str, float]:
        with contextlib.redirect_stdout(io.StringIO()):
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        dummy_id = answer_token_ids["N"]
        extended_tokens = prompt_tokens + [dummy_id]
        extended_input = types.ModelInput.from_ints(extended_tokens)
        topk_result = sampling_client.sample(
            prompt=extended_input,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=TOPK,
        ).result()
        topk_at_gen = (
            topk_result.topk_prompt_logprobs[-1]
            if topk_result.topk_prompt_logprobs
            else []
        )
        topk_lookup = {tid: lp for tid, lp in topk_at_gen} if topk_at_gen else {}
        found = {}
        for c in choices:
            lp = topk_lookup.get(answer_token_ids[c], None)
            if lp is not None:
                found[c] = lp
        if found:
            max_lp = max(found.values())
            exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
            total = sum(exps.values())
            return {c: exps.get(c, 0.0) / total for c in choices}
        return {c: 0.5 for c in choices}

    def _force_rollout(user_msg: str, thinking_text: str):
        segments = get_cumulative_cot_segments(thinking_text)
        n_sentences = len(segments)
        if n_sentences == 0:
            return []
        indices = list(range(0, n_sentences, SENTENCE_STRIDE))
        if (n_sentences - 1) not in indices:
            indices.append(n_sentences - 1)
        results = []
        for idx in indices:
            partial_cot = segments[idx]
            anchor = " So, the answer is: " if partial_cot else "So, the answer is: "
            cot_with_anchor = partial_cot + anchor
            prompt_str = (
                build_thinking_prompt(
                    tokenizer,
                    user_msg,
                    cot_prefix=cot_with_anchor,
                )
                + "</think>\n"
            )
            probs = _get_choice_probs(prompt_str)
            fraction_remaining = 1.0 - (idx + 1) / n_sentences
            results.append({
                "sentence_idx": idx,
                "fraction_remaining": fraction_remaining,
                "choice_probs": probs,
                "entropy": _shannon_entropy(probs),
            })
        return results

    # ── Build rollout list ────────────────────────────────────────────
    all_rollouts = []
    for _, row in all_df.iterrows():
        cot = row["cot_content"]
        if not isinstance(cot, str) or len(cot) < 50:
            continue
        qid = row["prompt_id"]
        all_rollouts.append({
            "prompt_id": qid,
            "rollout_idx": int(row["rollout_idx"]),
            "answer": str(row["answer"]),
            "label": row["label"],
            "user_msg": user_msgs[qid],
            "thinking": cot,
        })

    # ── Resume from checkpoint ────────────────────────────────────────
    checkpoint_path = OUTPUT_DIR / "forcing_checkpoint.json"
    done_keys = set()
    prior_results = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            prior_data = json.load(f)
        prior_results = prior_data.get("rollouts", [])
        for r in prior_results:
            done_keys.add((r["prompt_id"], r["rollout_idx"]))
        print(f"Resuming: {len(prior_results)} rollouts already done")

    remaining = [
        (i, r) for i, r in enumerate(all_rollouts)
        if (r["prompt_id"], r["rollout_idx"]) not in done_keys
    ]
    print(f"Remaining: {len(remaining)} rollouts to force")

    if not remaining:
        print("All rollouts already complete!")
        return

    # ── Run forcing ───────────────────────────────────────────────────
    new_results = []
    results_lock = threading.Lock()

    def _process_rollout(rollout):
        entropy_trajectory = _force_rollout(
            rollout["user_msg"], rollout["thinking"]
        )
        return {
            "prompt_id": rollout["prompt_id"],
            "rollout_idx": rollout["rollout_idx"],
            "answer": rollout["answer"],
            "label": rollout["label"],
            "n_sentences": len(entropy_trajectory),
            "results": entropy_trajectory,
        }

    n_workers = min(2000, len(remaining))
    n_errors = 0
    print(f"Forcing {len(remaining)} rollouts with {n_workers} parallel workers")
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_rollout, r): i
                for i, r in remaining
            }
            with tqdm(total=len(remaining), desc="Forcing rollouts") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        n_errors += 1
                        if n_errors <= 5:
                            print(f"\n  Error on rollout: {e}")
                        elif n_errors == 6:
                            print(f"\n  (suppressing further error messages)")
                        pbar.update(1)
                        continue
                    with results_lock:
                        new_results.append(result)
                        pbar.update(1)
                        if len(new_results) % 50 == 0:
                            _save_results(prior_results + new_results, checkpoint_path)
    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving {len(new_results)} new results...")
    finally:
        # Always save whatever we have
        all_results = prior_results + new_results
        _save_results(all_results, checkpoint_path)
        print(f"Checkpoint saved: {len(all_results)} total rollouts ({n_errors} errors)")

    if n_errors > 0:
        print(f"Warning: {n_errors} rollouts failed. Rerun --generate to retry.")

    all_results = prior_results + new_results
    run_dir = _create_run_dir("generate")
    final_path = run_dir / "forcing_results.json"
    _save_results(all_results, final_path)
    print(f"\nForcing complete. Saved {len(all_results)} rollouts to {final_path}")


def _save_results(results, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(
            {
                "model": SUBJECT_MODEL,
                "timestamp": datetime.now().isoformat(),
                "n_rollouts": len(results),
                "rollouts": results,
            },
            f,
            indent=2,
        )


# ── Phase 2: Train logistic regression baseline ──────────────────────


def _extract_entropy_features(trajectory: List[Dict]) -> np.ndarray:
    """Extract summary features from an entropy trajectory.

    Returns array of [mean, std, min, max, final, initial, slope].
    """
    if not trajectory:
        return np.zeros(7, dtype=np.float32)

    entropies = [r["entropy"] for r in trajectory]
    fracs = [r["fraction_remaining"] for r in trajectory]

    mean_ent = np.mean(entropies)
    std_ent = np.std(entropies)
    min_ent = np.min(entropies)
    max_ent = np.max(entropies)

    sorted_pairs = sorted(zip(fracs, entropies), reverse=True)
    initial_ent = sorted_pairs[0][1]
    final_ent = sorted_pairs[-1][1]

    if len(entropies) > 1:
        fracs_arr = np.array(fracs)
        ent_arr = np.array(entropies)
        slope = float(np.polyfit(fracs_arr, ent_arr, 1)[0])
    else:
        slope = 0.0

    return np.array(
        [mean_ent, std_ent, min_ent, max_ent, final_ent, initial_ent, slope],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "mean_entropy",
    "std_entropy",
    "min_entropy",
    "max_entropy",
    "final_entropy",
    "initial_entropy",
    "entropy_slope",
]


def train_baseline():
    """Train logistic regression on entropy features using LOO-CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Find forcing results
    latest_generate = OUTPUT_DIR / "latest_generate"
    if latest_generate.exists():
        results_path = latest_generate / "forcing_results.json"
    else:
        results_path = OUTPUT_DIR / "forcing_checkpoint.json"
    if not results_path.exists():
        print(f"No forcing results found in {OUTPUT_DIR}")
        print("Run with --generate first.")
        return
    print(f"Loading forcing results from {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    rollouts = data["rollouts"]
    print(f"Loaded {len(rollouts)} forced rollouts")

    # ── Extract features and labels ───────────────────────────────────
    X_list = []
    y_list = []
    prompt_ids = []
    metadata = []

    for rollout in rollouts:
        if not rollout["results"]:
            continue
        features = _extract_entropy_features(rollout["results"])
        # label: 1 = minority, 0 = majority
        label = 1 if rollout["label"] == "minority" else 0
        X_list.append(features)
        y_list.append(label)
        prompt_ids.append(rollout["prompt_id"])
        metadata.append({
            "prompt_id": rollout["prompt_id"],
            "rollout_idx": rollout["rollout_idx"],
            "answer": rollout["answer"],
            "label": rollout["label"],
        })

    X = np.stack(X_list)
    y = np.array(y_list)
    prompt_ids_arr = np.array(prompt_ids)
    print(f"Total samples: {len(y)}")
    print(f"  majority (0): {(y == 0).sum()}, minority (1): {(y == 1).sum()}")

    # Per-class metrics helper
    def _metrics(y_true, y_pred, pos_label=1):
        tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
        fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
        fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # ── LOO cross-validation ──────────────────────────────────────────
    folds = MinMajAnswerTask.loo_folds()
    fold_results = []
    all_test_preds = []
    all_test_true = []

    print(f"\n{'=' * 60}")
    print(f"  Entropy Baseline — LOO Cross-Validation ({len(folds)} folds)")
    print(f"{'=' * 60}")

    for fold in folds:
        test_pid_nm = fold["test_prompt_id"] + NM_SUFFIX
        train_pids_nm = {pid + NM_SUFFIX for pid in fold["train_prompt_ids"]}

        train_mask = np.isin(prompt_ids_arr, list(train_pids_nm))
        test_mask = prompt_ids_arr == test_pid_nm

        if not test_mask.any():
            print(f"  Fold {fold['fold_idx']} ({test_pid_nm}): no test data, skipping")
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Standardize per fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit logistic regression
        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        )
        clf.fit(X_train_s, y_train)

        test_pred = clf.predict(X_test_s)
        test_acc = float((y_test == test_pred).mean())
        min_m = _metrics(y_test, test_pred, pos_label=1)
        maj_m = _metrics(y_test, test_pred, pos_label=0)
        macro_f1 = (min_m["f1"] + maj_m["f1"]) / 2

        print(f"  Fold {fold['fold_idx']} ({fold['test_prompt_id']}): "
              f"Acc={test_acc:.3f}, F1={macro_f1:.3f} "
              f"(maj={maj_m['f1']:.3f}, min={min_m['f1']:.3f}), "
              f"n={len(y_test)}")

        fold_results.append({
            "fold_idx": fold["fold_idx"],
            "test_prompt_id": fold["test_prompt_id"],
            "accuracy": test_acc,
            "macro_f1": macro_f1,
            "majority_f1": maj_m["f1"],
            "majority_precision": maj_m["precision"],
            "majority_recall": maj_m["recall"],
            "minority_f1": min_m["f1"],
            "minority_precision": min_m["precision"],
            "minority_recall": min_m["recall"],
            "n_test": len(y_test),
            "n_train": len(y_train),
        })
        all_test_preds.extend(test_pred.tolist())
        all_test_true.extend(y_test.tolist())

    # ── Aggregate metrics ─────────────────────────────────────────────
    all_test_preds = np.array(all_test_preds)
    all_test_true = np.array(all_test_true)
    pooled_acc = float((all_test_true == all_test_preds).mean())
    pooled_min = _metrics(all_test_true, all_test_preds, pos_label=1)
    pooled_maj = _metrics(all_test_true, all_test_preds, pos_label=0)
    pooled_macro_f1 = (pooled_min["f1"] + pooled_maj["f1"]) / 2
    fold_f1s = [fr["macro_f1"] for fr in fold_results]
    mean_fold_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0

    majority_frac = float((all_test_true == 0).sum()) / len(all_test_true) if len(all_test_true) > 0 else 0.0

    print(f"\n  Aggregate ({len(fold_results)} folds):")
    print(f"    Pooled accuracy:    {pooled_acc:.3f} (n={len(all_test_true)})")
    print(f"    Pooled macro F1:    {pooled_macro_f1:.3f}")
    print(f"    Mean fold macro F1: {mean_fold_f1:.3f}")
    print(f"    Pooled majority F1: {pooled_maj['f1']:.3f}")
    print(f"    Pooled minority F1: {pooled_min['f1']:.3f}")
    print(f"    Chance baseline (always majority): {majority_frac:.3f}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "method": "loo_cv",
        "pooled_accuracy": pooled_acc,
        "pooled_macro_f1": pooled_macro_f1,
        "mean_fold_macro_f1": mean_fold_f1,
        "pooled_majority_f1": pooled_maj["f1"],
        "pooled_majority_precision": pooled_maj["precision"],
        "pooled_majority_recall": pooled_maj["recall"],
        "pooled_minority_f1": pooled_min["f1"],
        "pooled_minority_precision": pooled_min["precision"],
        "pooled_minority_recall": pooled_min["recall"],
        "chance_baseline": majority_frac,
        "n_folds": len(fold_results),
        "n_total": len(all_test_true),
        "feature_names": FEATURE_NAMES,
        "fold_results": fold_results,
    }
    run_dir = _create_run_dir("train")
    with open(run_dir / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {run_dir / 'baseline_results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Entropy baseline for majority/minority classification",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Phase 1: run forcing on rollouts (requires Tinker)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Phase 2: train logistic regression on entropy features",
    )
    args = parser.parse_args()

    if not args.generate and not args.train:
        parser.print_help()
        print("\nSpecify --generate, --train, or both.")
        return

    if args.generate:
        generate_forcing_data()

    if args.train:
        train_baseline()


if __name__ == "__main__":
    main()
