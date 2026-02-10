#!/usr/bin/env python3
"""
Data scaling analysis for sycophancy probe.

Trains the sycophancy attention probe with increasing fractions of training
data and evaluates test F1 at each size. Produces a plot of test F1 vs
training set size to check whether more data would help.

Usage:
    python -m src2.runs.run_scruples_data_scaling
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src2.methods.attention_probe import AttentionPoolingProbe
from src2.tasks import ScruplesTask
from src2.tasks.scruples.prompts import INTERVENTION_SUGGESTED_ANSWER

# ── Configuration (mirrors run_sycophancy_probe.py) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"

SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

VARIANTS = ["suggest_wrong", "suggest_right"]
NUM_CLASSES = 2

# Training hyperparameters
NUM_HEADS = 4
LR = 1e-4
EPOCHS = 40
BATCH_SIZE = 8
GRAD_CLIP = 1.0
TEST_SPLIT = 0.2
SWITCH_THRESHOLD = 0.40
HIGH_INTERVENTION_RATE = 0.82
LOW_INTERVENTION_RATE = 0.70
N_SYC_HIGH_PER_VARIANT = 25
N_SYC_LOW_PER_VARIANT = 25
N_NON_SYC_PER_VARIANT = 50
SEED = 42

# Data scaling fractions
DATA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 3  # repeat each fraction to get error bars


# ── Helpers ──────────────────────────────────────────────────────────


def _compute_f1(y_true, y_pred):
    """Compute F1 for class 1 (sycophantic)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def train_and_evaluate_single(train_X, train_y, test_X, test_y):
    """Train probe and return test F1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = train_X[0].shape[1]
    max_seq_len = max(
        max(x.shape[0] for x in train_X),
        max(x.shape[0] for x in test_X),
    )
    n_samples = len(train_X)

    class_counts = np.bincount(train_y, minlength=NUM_CLASSES)
    class_weights = n_samples / (NUM_CLASSES * class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    probe = AttentionPoolingProbe(
        hidden_dim=hidden_dim,
        num_heads=NUM_HEADS,
        output_dim=NUM_CLASSES,
        max_seq_len=max_seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    probe.train()
    for epoch in range(EPOCHS):
        perm = np.random.permutation(n_samples)
        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
            batch_idx = perm[start:end]
            batch_X = [train_X[i] for i in batch_idx]
            batch_y = train_y[batch_idx]
            batch_max_len = max(x.shape[0] for x in batch_X)

            X_pad = torch.zeros(len(batch_X), batch_max_len, hidden_dim, device=device)
            mask = torch.zeros(len(batch_X), batch_max_len, dtype=torch.bool, device=device)
            for i, x in enumerate(batch_X):
                sl = x.shape[0]
                X_pad[i, :sl, :] = torch.from_numpy(x).to(device)
                mask[i, :sl] = True
            y_t = torch.tensor(batch_y, dtype=torch.long, device=device)

            optimizer.zero_grad()
            pred = probe(X_pad, mask)
            loss = loss_fn(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            optimizer.step()

    # Evaluate
    probe.eval()
    preds = []
    for x in test_X:
        seq_len = x.shape[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = probe(x_t, m)
        preds.append(int(logits.argmax(dim=-1).item()))
    preds = np.array(preds)

    return _compute_f1(test_y, preds)


def train_test_split_by_anecdote(X_list, y, anecdote_ids, metadata,
                                  test_fraction=TEST_SPLIT, seed=SEED,
                                  anecdote_strata=None):
    """Split by anecdote with optional stratification (copied from run_sycophancy_probe)."""
    rng = np.random.default_rng(seed)

    if anecdote_strata is not None:
        strata_groups = {}
        for aid in set(anecdote_ids):
            s = anecdote_strata.get(aid, "unknown")
            strata_groups.setdefault(s, []).append(aid)

        train_anecdotes = set()
        test_anecdotes = set()
        for stratum, aids in sorted(strata_groups.items()):
            aids = sorted(aids)
            rng.shuffle(aids)
            n_test = max(1, int(len(aids) * test_fraction))
            test_anecdotes.update(aids[:n_test])
            train_anecdotes.update(aids[n_test:])
    else:
        unique_anecdotes = list(set(anecdote_ids))
        rng.shuffle(unique_anecdotes)
        n_test = max(1, int(len(unique_anecdotes) * test_fraction))
        test_anecdotes = set(unique_anecdotes[:n_test])
        train_anecdotes = set(unique_anecdotes[n_test:])

    train_idx = [i for i, a in enumerate(anecdote_ids) if a in train_anecdotes]
    test_idx = [i for i, a in enumerate(anecdote_ids) if a in test_anecdotes]

    return {
        "train_X": [X_list[i] for i in train_idx],
        "train_y": y[train_idx],
        "train_anecdote_ids": [anecdote_ids[i] for i in train_idx],
        "test_X": [X_list[i] for i in test_idx],
        "test_y": y[test_idx],
        "test_anecdote_ids": [anecdote_ids[i] for i in test_idx],
    }


def main():
    # ── Load data (same as run_sycophancy_probe) ──────────────────────
    tasks = {}
    for variant in VARIANTS:
        tasks[variant] = ScruplesTask(
            subject_model=SUBJECT_MODEL,
            variant=variant,
            data_dir=DATA_DIR,
        )

    task = tasks[VARIANTS[0]]
    print("Computing uncertainty-robust split...")
    split_info = task.get_uncertainty_robust_split(
        switch_threshold=SWITCH_THRESHOLD,
        non_syc_max_switch=0.10,
        high_intervention_rate=HIGH_INTERVENTION_RATE,
        low_intervention_rate=LOW_INTERVENTION_RATE,
        n_syc_high_per_variant=N_SYC_HIGH_PER_VARIANT,
        n_syc_low_per_variant=N_SYC_LOW_PER_VARIANT,
        n_non_syc_per_variant=N_NON_SYC_PER_VARIANT,
        variants=VARIANTS,
    )

    print("Loading sycophancy probe data...")
    probe_data = task.get_sycophancy_probe_data(
        variants=VARIANTS,
        layer=LAYER,
        data_slice=split_info["data_slice"],
        switch_threshold=SWITCH_THRESHOLD,
    )

    X_list_all = probe_data["X_list"]
    y_all = probe_data["y"]
    anecdote_ids_all = probe_data["anecdote_ids"]
    metadata_all = probe_data["metadata"]

    # Filter to intervention arm only (exclude control)
    intv_mask = [m["arm"] == "intervention" for m in metadata_all]
    X_list_intv = [x for x, keep in zip(X_list_all, intv_mask) if keep]
    y_intv = y_all[np.array(intv_mask)]
    anecdote_ids_intv = [a for a, keep in zip(anecdote_ids_all, intv_mask) if keep]
    metadata_intv = [m for m, keep in zip(metadata_all, intv_mask) if keep]

    print(f"Loaded {len(X_list_all)} total, kept {len(X_list_intv)} intervention-only")

    # Clean-example filter: keep only sycophantic runs from sycophantic
    # prompts, and control-majority-matching runs from non-sycophantic prompts
    clean_mask = []
    for m in metadata_intv:
        variant = m["variant"]
        syc_answer = INTERVENTION_SUGGESTED_ANSWER[variant]
        non_syc_answer = "B" if syc_answer == "A" else "A"
        ctrl_rate = m.get("control_sycophancy_rate", 0.0)
        majority_ctrl_answer = syc_answer if ctrl_rate > 0.5 else non_syc_answer
        answer = m.get("answer", "")
        if not isinstance(answer, str):
            clean_mask.append(False)
            continue

        if m["prompt_is_sycophantic"]:
            # Keep only runs where answer matches the suggested (sycophantic) answer
            clean_mask.append(m["is_sycophantic_answer"])
        else:
            # Keep only runs where answer matches the majority control answer
            clean_mask.append(answer.upper() == majority_ctrl_answer)

    X_list = [x for x, keep in zip(X_list_intv, clean_mask) if keep]
    y = y_intv[np.array(clean_mask)]
    anecdote_ids = [a for a, keep in zip(anecdote_ids_intv, clean_mask) if keep]
    metadata = [m for m, keep in zip(metadata_intv, clean_mask) if keep]

    print(f"  After clean-example filter: {len(X_list)} (discarded {len(X_list_intv) - len(X_list)})")
    print(f"  Class 0 (non_sycophantic): {(y == 0).sum()}")
    print(f"  Class 1 (sycophantic):     {(y == 1).sum()}")

    if len(X_list) < 10:
        print("Too few samples. Exiting.")
        return

    # ── Fixed train/test split ────────────────────────────────────────
    split = train_test_split_by_anecdote(
        X_list, y, anecdote_ids, metadata,
        anecdote_strata=split_info.get("anecdote_strata"),
    )
    full_train_X = split["train_X"]
    full_train_y = split["train_y"]
    test_X = split["test_X"]
    test_y = split["test_y"]
    n_train = len(full_train_X)
    print(f"Train: {n_train}, Test: {len(test_X)}")

    # ── Data scaling ──────────────────────────────────────────────────
    print("\nRunning data scaling analysis...")
    sizes = []
    mean_f1s = []
    std_f1s = []

    for frac in DATA_FRACTIONS:
        n_subset = max(2, int(n_train * frac))
        f1_scores = []

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(SEED + rep)
            indices = rng.choice(n_train, n_subset, replace=False)
            sub_X = [full_train_X[i] for i in indices]
            sub_y = full_train_y[indices]

            f1 = train_and_evaluate_single(sub_X, sub_y, test_X, test_y)
            f1_scores.append(f1)

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        sizes.append(n_subset)
        mean_f1s.append(mean_f1)
        std_f1s.append(std_f1)
        print(f"  {frac*100:5.1f}% ({n_subset:4d} samples): F1 = {mean_f1:.3f} ± {std_f1:.3f}")

    # ── Save results ──────────────────────────────────────────────────
    output_dir = DATA_DIR / "data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "fractions": DATA_FRACTIONS,
        "sizes": sizes,
        "mean_f1": mean_f1s,
        "std_f1": std_f1s,
        "n_repeats": N_REPEATS,
        "n_train_full": n_train,
        "n_test": len(test_X),
    }
    with open(output_dir / "scruples_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    from src2.utils.plotting import plot_data_scaling
    plot_data_scaling(
        sizes=sizes,
        scores=mean_f1s,
        metric_name="f1",
        output_path=output_dir / "scruples_data_scaling.png",
        title="Sycophancy Probe: Test F1 vs Training Data Size",
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
