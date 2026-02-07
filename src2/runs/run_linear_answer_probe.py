#!/usr/bin/env python3
"""
Linear probe diagnostic: last-token activation → soft A/B/C/D distribution.

Uses the last token of each forcing boundary (single [hidden_dim] vector),
per-question mean subtraction, and a simple Linear(hidden_dim, 4) probe.
Minimal capacity — can't memorize, tests whether the signal exists at all.

Usage:
    python -m src2.runs.run_linear_answer_probe
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src2.data_slice import DataSlice
from src2.tasks.forced_response.task import ForcingTask
from src2.utils.questions import load_gpqa_from_huggingface

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

CUSTOM_TRAIN_IDS = ["custom_bagel_001", "starfish", "waffle"]
NUM_GPQA_TRAIN = 50
NUM_GPQA_EVAL = 10
EXTRA_EVAL_IDS = ["blackmail_mc_001", "blackmail_ab_001"]

MAX_SENTENCES_PER_QUESTION_TRAIN = 15
MAX_SENTENCES_PER_QUESTION_EVAL = 20
ANSWER_LABELS = ["A", "B", "C", "D"]
NUM_CLASSES = 4

TOKEN_POSITION = "last_thinking"  # single vector per sample
MEAN_SUBTRACT = True

# Training
LR = 1e-3
EPOCHS = 200
WEIGHT_DECAY = 1e-2
SEED = 42

# Validation / early stopping
VAL_SPLIT = 0.2
PATIENCE = 20
MIN_DELTA = 0.005


# ── Helpers (shared with run_answer_probe) ────────────────────────────

def sample_sentence_indices(forcing_dir, question_ids, max_per_question, seed):
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


def dist_dict_to_array(dist_dict):
    vals = [dist_dict.get(label, 0.0) for label in ANSWER_LABELS]
    total = sum(vals)
    if total == 0:
        return None
    return np.array([v / total for v in vals], dtype=np.float32)


def load_probe_data(forcing, question_ids, sentence_map, layer, token_position):
    X_list, y_list, q_ids, s_indices = [], [], [], []
    for qid in question_ids:
        ds = DataSlice(ids={qid}, sentence_indices=sentence_map.get(qid))
        samples = forcing.get_probe_data(layer, ds, token_position)
        for sample in samples:
            dist = dist_dict_to_array(sample["answer_distribution"])
            if dist is None:
                continue
            act = sample["activation"]
            if act.ndim != 1:
                continue  # skip if not single-vector (shouldn't happen with last_thinking)
            X_list.append(act)
            y_list.append(dist)
            q_ids.append(sample["question_id"])
            s_indices.append(sample["sentence_idx"])
    X = np.stack(X_list) if X_list else np.zeros((0, 5120))
    y = np.stack(y_list) if y_list else np.zeros((0, NUM_CLASSES))
    return {"X": X, "y": y, "question_ids": q_ids, "sentence_indices": s_indices}


def mean_subtract(data):
    """Subtract per-question mean activation."""
    qid_to_idx = defaultdict(list)
    for i, qid in enumerate(data["question_ids"]):
        qid_to_idx[qid].append(i)
    means = {}
    for qid, indices in qid_to_idx.items():
        means[qid] = data["X"][indices].mean(axis=0)
    for i, qid in enumerate(data["question_ids"]):
        data["X"][i] -= means[qid]
    return data


def build_question_splits(forcing_dir, seed=SEED):
    rng = np.random.default_rng(seed)
    gpqa_questions = load_gpqa_from_huggingface(
        subset="gpqa_diamond", max_questions=NUM_GPQA_TRAIN + NUM_GPQA_EVAL,
    )
    gpqa_ids = [q.id for q in gpqa_questions]
    available_ids = [qid for qid in gpqa_ids if (forcing_dir / qid).exists()]
    rng.shuffle(available_ids)
    gpqa_train = available_ids[:NUM_GPQA_TRAIN]
    gpqa_eval = available_ids[NUM_GPQA_TRAIN:NUM_GPQA_TRAIN + NUM_GPQA_EVAL]
    train_ids = CUSTOM_TRAIN_IDS + gpqa_train
    eval_ids = gpqa_eval + [
        eid for eid in EXTRA_EVAL_IDS if (forcing_dir / eid).exists()
    ]
    return {"train_ids": train_ids, "eval_ids": eval_ids}


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, question_ids):
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0}

    true_hard = y_true.argmax(axis=1)
    pred_hard = y_pred.argmax(axis=1)
    hard_acc = float((true_hard == pred_hard).mean())

    class_counts = Counter(true_hard.tolist())
    chance = max(class_counts.values()) / n

    eps = 1e-8
    pred_c = np.clip(y_pred, eps, 1.0)
    true_c = np.clip(y_true, eps, 1.0)
    soft_ce = float(-(true_c * np.log(pred_c)).sum(axis=1).mean())
    kl = float((true_c * np.log(true_c / pred_c)).sum(axis=1).mean())

    per_q = {}
    for qid in sorted(set(question_ids)):
        idx = [i for i, q in enumerate(question_ids) if q == qid]
        q_true = true_hard[idx]
        q_pred = pred_hard[idx]
        per_q[qid] = {
            "n": len(idx),
            "acc": float((q_true == q_pred).mean()),
        }

    return {
        "hard_accuracy": hard_acc,
        "chance_baseline": chance,
        "soft_ce": soft_ce,
        "kl_divergence": kl,
        "n_samples": n,
        "per_question": per_q,
    }


def print_metrics(metrics, label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if metrics.get("n_samples", 0) == 0:
        print("  No samples.")
        return
    print(f"  Argmax accuracy: {metrics['hard_accuracy']:.3f}  (chance: {metrics['chance_baseline']:.3f})")
    print(f"  Soft CE loss:    {metrics['soft_ce']:.4f}")
    print(f"  KL divergence:   {metrics['kl_divergence']:.4f}")
    print(f"  N samples:       {metrics['n_samples']}")
    if "per_question" in metrics:
        print(f"\n  Per-question:")
        for qid, v in metrics["per_question"].items():
            print(f"    {qid:>30s}: acc={v['acc']:.3f}  n={v['n']}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # Splits
    splits = build_question_splits(forcing.forcing_dir)
    train_ids = splits["train_ids"]
    eval_ids = splits["eval_ids"]
    print(f"Train: {len(train_ids)} questions, Eval: {len(eval_ids)} questions")

    # Sentence sampling
    train_smap = sample_sentence_indices(
        forcing.forcing_dir, train_ids, MAX_SENTENCES_PER_QUESTION_TRAIN, SEED,
    )
    eval_smap = sample_sentence_indices(
        forcing.forcing_dir, eval_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    smap = {**train_smap, **eval_smap}

    # Train/val split by question
    rng = np.random.default_rng(SEED)
    shuffled = list(train_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_SPLIT))
    val_ids = shuffled[:n_val]
    actual_train_ids = shuffled[n_val:]
    print(f"Train/val split: {len(actual_train_ids)} train, {len(val_ids)} val")

    # Load data (single vector per sample)
    print(f"\nLoading data (token_position={TOKEN_POSITION})...")
    train_data = load_probe_data(forcing, actual_train_ids, smap, LAYER, TOKEN_POSITION)
    val_data = load_probe_data(forcing, val_ids, smap, LAYER, TOKEN_POSITION)
    eval_data = load_probe_data(forcing, eval_ids, smap, LAYER, TOKEN_POSITION)
    print(f"  Train: {train_data['X'].shape[0]}, Val: {val_data['X'].shape[0]}, Eval: {eval_data['X'].shape[0]}")

    # Mean subtraction
    if MEAN_SUBTRACT:
        print("Applying per-question mean subtraction...")
        train_data = mean_subtract(train_data)
        val_data = mean_subtract(val_data)
        eval_data = mean_subtract(eval_data)

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data["X"])
    X_val = scaler.transform(val_data["X"])
    X_eval = scaler.transform(eval_data["X"])
    y_train = train_data["y"]
    y_val = val_data["y"]
    y_eval = eval_data["y"]

    # Train linear probe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = X_train.shape[1]
    probe = torch.nn.Linear(hidden_dim, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"\nTraining linear probe: Linear({hidden_dim}→{NUM_CLASSES}), {n_params:,} params")

    optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    def soft_ce(logits, targets):
        return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        # Train
        probe.train()
        optimizer.zero_grad()
        loss = soft_ce(probe(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        # Val
        probe.eval()
        with torch.no_grad():
            val_loss = soft_ce(probe(X_val_t), y_val_t).item()
            train_loss = loss.item()

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                train_acc = (probe(X_train_t).argmax(-1) == y_train_t.argmax(-1)).float().mean().item()
                val_acc = (probe(X_val_t).argmax(-1) == y_val_t.argmax(-1)).float().mean().item()
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS}  "
                f"train: {train_loss:.4f} (acc {train_acc:.3f})  "
                f"val: {val_loss:.4f} (acc {val_acc:.3f})  "
                f"best: {best_val_loss:.4f} (ep {best_epoch})"
            )

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        probe.load_state_dict(best_state)
        probe.to(device)
        print(f"  Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # Predict
    probe.eval()
    with torch.no_grad():
        X_eval_t = torch.from_numpy(X_eval).float().to(device)
        train_pred = torch.softmax(probe(X_train_t), -1).cpu().numpy()
        val_pred = torch.softmax(probe(X_val_t), -1).cpu().numpy()
        eval_pred = torch.softmax(probe(X_eval_t), -1).cpu().numpy()

    # Metrics
    print_metrics(compute_metrics(y_train, train_pred, train_data["question_ids"]), "Train")
    print_metrics(compute_metrics(y_val, val_pred, val_data["question_ids"]), "Validation")
    print_metrics(compute_metrics(y_eval, eval_pred, eval_data["question_ids"]), "Eval (held-out)")

    # Save
    output_dir = DATA_DIR / "linear_answer_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    class _Enc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output = {
        "train_metrics": compute_metrics(y_train, train_pred, train_data["question_ids"]),
        "val_metrics": compute_metrics(y_val, val_pred, val_data["question_ids"]),
        "eval_metrics": compute_metrics(y_eval, eval_pred, eval_data["question_ids"]),
        "config": {
            "layer": LAYER,
            "token_position": TOKEN_POSITION,
            "mean_subtract": MEAN_SUBTRACT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "best_epoch": best_epoch,
            "n_params": n_params,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_eval": len(X_eval),
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2, cls=_Enc)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
