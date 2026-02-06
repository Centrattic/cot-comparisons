#!/usr/bin/env python3
"""
Soft A/B/C/D attention probe.

Trains a generalizable attention probe that predicts soft answer probability
distributions (A/B/C/D) from model activations at forcing boundaries.

Uses soft cross-entropy loss against the actual answer_distribution from
forcing, rather than hard argmax labels.

To avoid over-representing questions with long CoTs, randomly samples a
fixed number of sentence indices per question/rollout for activation
extraction and training.

Usage:
    python -m src2.runs.run_answer_probe
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src2.data_slice import DataSlice
from src2.methods.attention_probe import AttentionPoolingProbe
from src2.tasks.forced_response.task import ForcingTask
from src2.utils.questions import load_gpqa_from_huggingface

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

# Custom questions always go to train
CUSTOM_TRAIN_IDS = ["custom_bagel_001", "starfish", "waffle"]

# GPQA Diamond: load N total, split into train/eval
NUM_GPQA_TRAIN = 50
NUM_GPQA_EVAL = 10

# Extra eval-only questions
EXTRA_EVAL_IDS = ["blackmail_mc_001", "blackmail_ab_001"]

# Sentence sampling: sample this many sentence indices per question.
# Each index appears in all rollouts, so total samples per question ≈
# MAX_SENTENCES_PER_QUESTION × num_rollouts.
# With 10 rollouts: 20 sentences × 10 rollouts = 200 samples/question.
# With 53 train questions: ~10,600 training samples.
MAX_SENTENCES_PER_QUESTION = 20
ANSWER_LABELS = ["A", "B", "C", "D"]
NUM_CLASSES = 4

# Training hyperparameters
NUM_HEADS = 4
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 32
SEED = 42

EXTRACT_ACTIVATIONS = True
TOKEN_POSITION = "full_sequence"


# ── Helpers ───────────────────────────────────────────────────────────


def sample_sentence_indices(
    forcing_dir: Path,
    question_ids: list,
    max_per_question: int,
    seed: int,
) -> dict:
    """For each question, find available sentence indices and randomly sample N.

    Returns:
        {qid: set(int)} mapping question IDs to sampled sentence indices.
    """
    rng = np.random.default_rng(seed)
    sentence_map = {}
    for qid in question_ids:
        summaries = sorted(
            forcing_dir.glob(f"{qid}/rollout_*/*/sentence_*/summary.json")
        )
        available = {int(p.parent.name.split("_")[1]) for p in summaries}
        if len(available) <= max_per_question:
            sentence_map[qid] = available
        else:
            sentence_map[qid] = set(
                rng.choice(sorted(available), max_per_question, replace=False)
            )
    return sentence_map


def dist_dict_to_array(dist_dict: dict) -> np.ndarray:
    """Convert {"A": 0.4, "B": 0.28, ...} → np.array([0.4, 0.28, 0.13, 0.19]).

    Only includes A/B/C/D keys. Re-normalizes to sum to 1.
    Returns None if no ABCD keys are present.
    """
    vals = [dist_dict.get(label, 0.0) for label in ANSWER_LABELS]
    total = sum(vals)
    if total == 0:
        return None
    return np.array([v / total for v in vals], dtype=np.float32)


def load_probe_data(
    forcing: ForcingTask,
    question_ids: list,
    sentence_map: dict,
    layer: int,
    token_position: str,
) -> dict:
    """Load activations + soft distributions for the given questions.

    Uses per-question DataSlice to respect sampled sentences.

    Returns:
        {
            "X_list": list of ndarrays,
            "y_soft": ndarray [n, 4],
            "question_ids": list,
            "sentence_indices": list,
        }
    """
    X_list = []
    y_soft_list = []
    q_ids = []
    s_indices = []

    for qid in question_ids:
        ds = DataSlice(ids={qid}, sentence_indices=sentence_map.get(qid))
        samples = forcing.get_probe_data(layer, ds, token_position)

        for sample in samples:
            dist = dist_dict_to_array(sample["answer_distribution"])
            if dist is None:
                continue
            X_list.append(sample["activation"])
            y_soft_list.append(dist)
            q_ids.append(sample["question_id"])
            s_indices.append(sample["sentence_idx"])

    y_soft = np.stack(y_soft_list) if y_soft_list else np.zeros((0, NUM_CLASSES))

    return {
        "X_list": X_list,
        "y_soft": y_soft,
        "question_ids": q_ids,
        "sentence_indices": s_indices,
    }


def soft_cross_entropy(
    logits: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy: -(target * log_softmax(logits)).sum(-1).mean()"""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


# ── Training ──────────────────────────────────────────────────────────


def train_and_evaluate(
    train_X: list,
    train_y_soft: np.ndarray,
    test_X: list,
    test_y_soft: np.ndarray,
    num_heads: int = NUM_HEADS,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train probe with soft CE loss on train set, evaluate on test set.

    Returns dict with predictions, true labels, and final train loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}, batch_size: {batch_size}")

    hidden_dim = train_X[0].shape[1]
    max_seq_len = max(
        max(x.shape[0] for x in train_X),
        max(x.shape[0] for x in test_X) if test_X else 0,
    )
    n_samples = len(train_X)

    probe = AttentionPoolingProbe(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        output_dim=NUM_CLASSES,
        max_seq_len=max_seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    probe.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]

            batch_X = [train_X[i] for i in batch_idx]
            batch_y = train_y_soft[batch_idx]
            batch_max_len = max(x.shape[0] for x in batch_X)

            X_pad = torch.zeros(len(batch_X), batch_max_len, hidden_dim, device=device)
            mask = torch.zeros(
                len(batch_X), batch_max_len, dtype=torch.bool, device=device
            )
            for i, x in enumerate(batch_X):
                sl = x.shape[0]
                X_pad[i, :sl, :] = torch.from_numpy(x).to(device)
                mask[i, :sl] = True
            y_t = torch.from_numpy(batch_y).float().to(device)

            optimizer.zero_grad()
            pred = probe(X_pad, mask)
            loss = soft_cross_entropy(pred, y_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / n_batches:.4f}")

    # Evaluate on test set (may be empty)
    probe.eval()
    test_pred_dists = []
    if test_X:
        for x in test_X:
            seq_len = x.shape[0]
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
            m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            with torch.no_grad():
                logits = probe(x_t, m)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
            test_pred_dists.append(probs.cpu().numpy())

    # Also get train predictions for metrics
    train_pred_dists = []
    for x in train_X:
        seq_len = x.shape[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = probe(x_t, m)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        train_pred_dists.append(probs.cpu().numpy())

    return {
        "test_pred_dists": np.array(test_pred_dists)
        if test_pred_dists
        else np.zeros((0, NUM_CLASSES)),
        "train_pred_dists": np.array(train_pred_dists),
        "final_train_loss": float(epoch_loss / max(n_batches, 1)),
    }


# ── Metrics ───────────────────────────────────────────────────────────


def compute_metrics(
    y_true_soft: np.ndarray,
    y_pred_soft: np.ndarray,
    question_ids: list,
) -> dict:
    """Compute hard accuracy, mean KL divergence, and per-question breakdown.

    Args:
        y_true_soft: [n, 4] true soft distributions
        y_pred_soft: [n, 4] predicted soft distributions
        question_ids: list of question IDs for per-question breakdown
    """
    n = len(y_true_soft)
    if n == 0:
        return {"n_samples": 0}

    # Hard accuracy (argmax match)
    true_hard = y_true_soft.argmax(axis=1)
    pred_hard = y_pred_soft.argmax(axis=1)
    hard_accuracy = (true_hard == pred_hard).mean()

    # Chance baseline
    class_counts = Counter(true_hard.tolist())
    chance_baseline = max(class_counts.values()) / n if n > 0 else 0.0

    # Mean KL divergence: KL(true || pred)
    eps = 1e-8
    pred_clipped = np.clip(y_pred_soft, eps, 1.0)
    true_clipped = np.clip(y_true_soft, eps, 1.0)
    kl_per_sample = (true_clipped * np.log(true_clipped / pred_clipped)).sum(axis=1)
    mean_kl = float(kl_per_sample.mean())

    # Confusion matrix (hard labels): rows = true, cols = predicted
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(true_hard, pred_hard):
        conf_matrix[t, p] += 1

    # Per-question breakdown
    per_question = {}
    unique_qids = sorted(set(question_ids))
    for qid in unique_qids:
        idx = [i for i, q in enumerate(question_ids) if q == qid]
        q_true = true_hard[idx]
        q_pred = pred_hard[idx]
        q_kl = kl_per_sample[idx]
        per_question[qid] = {
            "n_samples": len(idx),
            "hard_accuracy": float((q_true == q_pred).mean()),
            "mean_kl": float(q_kl.mean()),
        }

    return {
        "hard_accuracy": float(hard_accuracy),
        "chance_baseline": float(chance_baseline),
        "mean_kl_divergence": mean_kl,
        "n_samples": n,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_labels": ANSWER_LABELS,
        "per_question": per_question,
    }


def print_results(metrics: dict, label: str = "Results"):
    """Pretty-print metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    if metrics.get("n_samples", 0) == 0:
        print("  No samples.")
        return

    print(f"  Hard accuracy:   {metrics['hard_accuracy']:.3f}")
    print(f"  Chance baseline: {metrics['chance_baseline']:.3f}")
    print(f"  Mean KL div:     {metrics['mean_kl_divergence']:.4f}")
    print(f"  N samples:       {metrics['n_samples']}")

    if "per_question" in metrics:
        print(f"\n  Per-question breakdown:")
        for qid, vals in metrics["per_question"].items():
            print(
                f"    {qid:>25s}: "
                f"acc={vals['hard_accuracy']:.3f}  "
                f"KL={vals['mean_kl']:.4f}  "
                f"n={vals['n_samples']}"
            )

    if "confusion_matrix" in metrics:
        print(f"\n  Confusion matrix (rows=true, cols=predicted):")
        labels = metrics["confusion_labels"]
        header = "            " + "  ".join(f"{l:>6s}" for l in labels)
        print(header)
        for i, row in enumerate(metrics["confusion_matrix"]):
            row_str = "  ".join(f"{v:>6d}" for v in row)
            print(f"    {labels[i]:>6s}  {row_str}")


# ── Main ──────────────────────────────────────────────────────────────


def build_question_splits(forcing_dir: Path, seed: int = SEED) -> dict:
    """Build train/eval question ID lists from custom + GPQA Diamond questions.

    Loads GPQA Diamond from HuggingFace, deterministically shuffles, and
    splits into train/eval. Only includes questions that have forcing data
    on disk (skips questions that haven't been forced yet).

    Returns:
        {"train_ids": list[str], "eval_ids": list[str],
         "n_gpqa_available": int, "n_gpqa_skipped": int}
    """
    rng = np.random.default_rng(seed)

    # Load all GPQA Diamond questions
    gpqa_questions = load_gpqa_from_huggingface(
        subset="gpqa_diamond", max_questions=NUM_GPQA_TRAIN + NUM_GPQA_EVAL,
    )
    gpqa_ids = [q.id for q in gpqa_questions]

    # Filter to only questions with forcing data on disk
    available_ids = [
        qid for qid in gpqa_ids if (forcing_dir / qid).exists()
    ]
    skipped = len(gpqa_ids) - len(available_ids)

    # Deterministic shuffle then split
    available_ids = list(available_ids)
    rng.shuffle(available_ids)
    gpqa_train = available_ids[:NUM_GPQA_TRAIN]
    gpqa_eval = available_ids[NUM_GPQA_TRAIN : NUM_GPQA_TRAIN + NUM_GPQA_EVAL]

    train_ids = CUSTOM_TRAIN_IDS + gpqa_train
    eval_ids = gpqa_eval + [
        eid for eid in EXTRA_EVAL_IDS if (forcing_dir / eid).exists()
    ]

    return {
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "n_gpqa_available": len(available_ids),
        "n_gpqa_skipped": skipped,
    }


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # ── Step 1: Build question splits ─────────────────────────────────
    print("Building question splits from custom + GPQA Diamond...")
    splits = build_question_splits(forcing.forcing_dir)
    train_ids = splits["train_ids"]
    eval_ids = splits["eval_ids"]

    print(f"  GPQA Diamond available on disk: {splits['n_gpqa_available']}")
    if splits["n_gpqa_skipped"] > 0:
        print(f"  GPQA Diamond not yet forced (skipped): {splits['n_gpqa_skipped']}")
    print(f"  Train questions: {len(train_ids)} ({len(CUSTOM_TRAIN_IDS)} custom + {len(train_ids) - len(CUSTOM_TRAIN_IDS)} GPQA)")
    print(f"  Eval questions:  {len(eval_ids)}")

    all_question_ids = train_ids + eval_ids

    # ── Step 2: Sample sentence indices ───────────────────────────────
    print("\nSampling sentence indices...")
    sentence_map = sample_sentence_indices(
        forcing.forcing_dir,
        all_question_ids,
        MAX_SENTENCES_PER_QUESTION,
        SEED,
    )
    total_sentences = sum(len(v) for v in sentence_map.values())
    print(f"  {total_sentences} unique sentence indices across {len(sentence_map)} questions")

    # ── Step 3: Extract activations if needed ─────────────────────────
    if EXTRACT_ACTIVATIONS:
        # No sentence_indices filter: batched extraction runs one forward pass
        # per rollout covering ALL sentences. Sentence sampling happens at
        # data-loading time (step 4).
        ds = DataSlice(ids=set(all_question_ids))
        print(f"\nExtracting activations for {len(all_question_ids)} questions "
              f"(all sentences per rollout, batched)...")
        forcing.extract_activations_batched(
            model_name=ACTIVATION_MODEL,
            layer=LAYER,
            data_slice=ds,
        )

    # ── Step 4: Load data ─────────────────────────────────────────────
    print("\nLoading training data...")
    train_data = load_probe_data(
        forcing, train_ids, sentence_map, LAYER, TOKEN_POSITION,
    )
    print(f"  Train: {len(train_data['X_list'])} samples from {len(train_ids)} questions")

    eval_data = None
    if eval_ids:
        print("Loading eval data...")
        eval_data = load_probe_data(
            forcing, eval_ids, sentence_map, LAYER, TOKEN_POSITION,
        )
        print(f"  Eval: {len(eval_data['X_list'])} samples from {len(eval_ids)} questions")

    if len(train_data["X_list"]) < 5:
        print("Too few training samples. Exiting.")
        return

    # ── Step 5: Train and evaluate ────────────────────────────────────
    print("\nTraining soft-label attention probe...")
    results = train_and_evaluate(
        train_X=train_data["X_list"],
        train_y_soft=train_data["y_soft"],
        test_X=eval_data["X_list"] if eval_data else [],
        test_y_soft=eval_data["y_soft"] if eval_data else np.zeros((0, NUM_CLASSES)),
    )

    # ── Step 6: Compute and print metrics ─────────────────────────────
    train_metrics = compute_metrics(
        train_data["y_soft"],
        results["train_pred_dists"],
        train_data["question_ids"],
    )
    print_results(train_metrics, label="Train Set Results")

    eval_metrics = {}
    if eval_data and len(eval_data["X_list"]) > 0:
        eval_metrics = compute_metrics(
            eval_data["y_soft"],
            results["test_pred_dists"],
            eval_data["question_ids"],
        )
        print_results(eval_metrics, label="Eval Set Results")

    # ── Step 7: Save results ──────────────────────────────────────────
    output_dir = DATA_DIR / "answer_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "config": {
            "subject_model": SUBJECT_MODEL,
            "activation_model": ACTIVATION_MODEL,
            "layer": LAYER,
            "token_position": TOKEN_POSITION,
            "train_question_ids": train_ids,
            "eval_question_ids": eval_ids,
            "max_sentences_per_question": MAX_SENTENCES_PER_QUESTION,
            "num_gpqa_train": NUM_GPQA_TRAIN,
            "num_gpqa_eval": NUM_GPQA_EVAL,
            "num_heads": NUM_HEADS,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "n_train_samples": len(train_data["X_list"]),
            "n_eval_samples": len(eval_data["X_list"]) if eval_data else 0,
            "sentence_map": {
                qid: sorted(indices) for qid, indices in sentence_map.items()
            },
        },
        "final_train_loss": results["final_train_loss"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
