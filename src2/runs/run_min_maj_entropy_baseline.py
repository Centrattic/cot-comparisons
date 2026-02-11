#!/usr/bin/env python3
"""
Entropy baseline for majority/minority answer classification.

Tests whether forced-answer entropy alone can distinguish majority
from minority rollouts. Two-phase script:

  Phase 1 (GENERATE): Run forced-response on rollouts to get entropy
    trajectories at each sentence boundary. Requires Tinker.

  Phase 2 (TRAIN): Load the saved entropy data, extract features
    (mean, final, std, etc.), and train logistic regression to
    classify majority vs minority. Reports accuracy/F1.

All prompts use A/B answer format from verification_rollouts.

Train prompts:
  bagel_ab, gpqa_nmr_compound_ab, gpqa_benzene_naming_ab,
  harder_well_ab, bookworm_ab

Test prompts:
  gpqa_diels_alder_ab, gpqa_optical_activity_ab

Usage:
    # Phase 1: generate forcing data (requires Tinker)
    python -m src2.runs.run_min_maj_entropy_baseline --generate

    # Phase 2: train baseline (no Tinker needed)
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

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
OUTPUT_DIR = DATA_DIR / "entropy_baseline"
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"
QUESTIONS_FILE = PROJECT_ROOT / "src2" / "utils" / "questions.json"

SUBJECT_MODEL = "Qwen/Qwen3-32B"

TRAIN_PROMPT_IDS = [
    "bagel_ab",
    "gpqa_nmr_compound_ab",
    "gpqa_benzene_naming_ab",
    "harder_well_ab",
    "bookworm_ab",
]

TEST_PROMPT_IDS = [
    "gpqa_diels_alder_ab",
    "gpqa_optical_activity_ab",
]

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
    all_prompt_ids = TRAIN_PROMPT_IDS + TEST_PROMPT_IDS
    dfs = []
    user_msgs = {}
    for qid in all_prompt_ids:
        df = _load_verification_rollouts(qid)
        dfs.append(df)
        user_msgs[qid] = _build_user_msg(qid)

    all_df = pd.concat(dfs, ignore_index=True)

    train_df = all_df[all_df["prompt_id"].isin(TRAIN_PROMPT_IDS)]
    test_df = all_df[all_df["prompt_id"].isin(TEST_PROMPT_IDS)]
    print(f"\nTrain: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"Test:  {len(test_df)} ({test_df['label'].value_counts().to_dict()})")
    print(f"Total rollouts to force: {len(all_df)}")

    # ── Resolve A/B token IDs ─────────────────────────────────────────
    # All prompts use A/B answer format
    choices = ["A", "B"]
    answer_token_ids = {}
    for ans in choices:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = tokenizer.encode(ans, add_special_tokens=False)
        answer_token_ids[ans] = ids[-1]
    print(f"Answer token IDs: {answer_token_ids}")

    assert answer_token_ids["A"] != answer_token_ids["B"], \
        "A and B map to the same token ID!"

    def _get_choice_probs(prompt_str: str) -> Dict[str, float]:
        with contextlib.redirect_stdout(io.StringIO()):
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        dummy_id = answer_token_ids["A"]
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
    print(f"Forcing {len(remaining)} rollouts with {n_workers} parallel workers")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_rollout, r): i
            for i, r in remaining
        }
        with tqdm(total=len(remaining), desc="Forcing rollouts") as pbar:
            for future in as_completed(futures):
                result = future.result()
                with results_lock:
                    new_results.append(result)
                    pbar.update(1)
                    if len(new_results) % 50 == 0:
                        _save_results(prior_results + new_results, checkpoint_path)

    all_results = prior_results + new_results
    _save_results(all_results, checkpoint_path)

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
    """Train logistic regression on entropy features."""
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
    print(f"Total samples: {len(y)}")
    print(f"  majority (0): {(y == 0).sum()}, minority (1): {(y == 1).sum()}")

    # ── Train/test split by prompt ID ─────────────────────────────────
    train_set = set(TRAIN_PROMPT_IDS)
    test_set = set(TEST_PROMPT_IDS)

    train_idx = [i for i, pid in enumerate(prompt_ids) if pid in train_set]
    test_idx = [i for i, pid in enumerate(prompt_ids) if pid in test_set]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(
        f"Train: {len(X_train)} ({y_train.sum()} minority), "
        f"Test: {len(X_test)} ({y_test.sum()} minority)"
    )

    # ── Standardize ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Logistic regression ───────────────────────────────────────────
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(X_train_s, y_train)

    train_pred = clf.predict(X_train_s)
    test_pred = clf.predict(X_test_s)

    train_acc = float((y_train == train_pred).mean())
    test_acc = float((y_test == test_pred).mean())

    # Per-class metrics
    def _metrics(y_true, y_pred, pos_label=1):
        tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
        fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
        fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    train_min_metrics = _metrics(y_train, train_pred, pos_label=1)
    test_min_metrics = _metrics(y_test, test_pred, pos_label=1)

    print(f"\n{'=' * 60}")
    print(f"  Entropy Baseline — Majority/Minority Classification")
    print(f"{'=' * 60}")
    print(f"  Train: Acc={train_acc:.3f}, Minority F1={train_min_metrics['f1']:.3f}")
    print(f"  Test:  Acc={test_acc:.3f}, Minority F1={test_min_metrics['f1']:.3f}")
    majority_frac = (y_test == 0).sum() / len(y_test) if len(y_test) > 0 else 0.0
    print(f"  Chance baseline (always majority): {majority_frac:.3f}")

    # Per-prompt test breakdown
    for pid in TEST_PROMPT_IDS:
        pid_idx = [i for i in test_idx if prompt_ids[i] == pid]
        if not pid_idx:
            continue
        pid_y = y[pid_idx]
        pid_pred = clf.predict(scaler.transform(X[pid_idx]))
        pid_acc = float((pid_y == pid_pred).mean())
        pid_m = _metrics(pid_y, pid_pred)
        print(f"\n  {pid} (n={len(pid_idx)}, {(pid_y == 1).sum()} minority):")
        print(f"    Acc={pid_acc:.3f}, Minority F1={pid_m['f1']:.3f}")

    # Feature importances
    print(f"\n  Feature coefficients:")
    for name, coef in sorted(
        zip(FEATURE_NAMES, clf.coef_[0]), key=lambda x: -abs(x[1])
    ):
        print(f"    {name:>20s}: {coef:+.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_minority_f1": train_min_metrics["f1"],
        "test_minority_f1": test_min_metrics["f1"],
        "test_minority_precision": test_min_metrics["precision"],
        "test_minority_recall": test_min_metrics["recall"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": FEATURE_NAMES,
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "train_prompt_ids": TRAIN_PROMPT_IDS,
        "test_prompt_ids": TEST_PROMPT_IDS,
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
