#!/usr/bin/env python3
"""
Intervention-type attention probe (confound test).

Trains a 3-class attention probe that predicts which intervention was applied
(none / suggest_wrong / suggest_right) from CoT + response activations only.

If this probe succeeds, it demonstrates that a sycophancy probe could be
exploiting intervention-type signal rather than detecting genuine sycophantic
reasoning.

Usage:
    python -m src2.runs.run_intervention_probe
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src2.tasks import ScruplesTask
from src2.methods import AttentionProbe
from src2.methods.attention_probe import AttentionPoolingProbe
from src2.data_slice import DataSlice

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"

SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32
LOAD_IN_4BIT = False

VARIANTS = ["suggest_wrong", "suggest_right"]
CLASS_NAMES = {0: "control", 1: "suggest_wrong", 2: "suggest_right"}
NUM_CLASSES = 3

# Training hyperparameters
NUM_HEADS = 4
LR = 1e-3
EPOCHS = 100

# Which steps to run
EXTRACT_ACTIVATIONS = True
# ──────────────────────────────────────────────────────────────────────


def leave_one_anecdote_out_cv(
    X_list: list,
    y: np.ndarray,
    anecdote_ids: list,
    num_heads: int = NUM_HEADS,
    lr: float = LR,
    epochs: int = EPOCHS,
) -> dict:
    """
    Leave-one-anecdote-out cross-validation for 3-class attention probe.

    Groups samples by anecdote_id, holds out all runs from one anecdote at
    a time, trains on the rest, and predicts on the held-out runs.
    """
    unique_anecdotes = sorted(set(anecdote_ids))
    anecdote_arr = np.array(anecdote_ids)

    all_preds = np.zeros(len(y), dtype=np.int64)
    all_probs = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)

    for fold_idx, held_out_aid in enumerate(unique_anecdotes):
        test_mask = anecdote_arr == held_out_aid
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(train_indices) < 5 or len(test_indices) == 0:
            continue

        # Check that train set has all classes
        train_y = y[train_indices]
        if len(set(train_y.tolist())) < NUM_CLASSES:
            # Skip fold if a class is missing from training
            continue

        train_X = [X_list[i] for i in train_indices]
        test_X = [X_list[i] for i in test_indices]
        test_y = y[test_indices]

        # Pad and tensorize train set
        hidden_dim = train_X[0].shape[1]
        max_train_len = max(x.shape[0] for x in train_X)
        X_train_pad = torch.zeros(len(train_X), max_train_len, hidden_dim)
        mask_train = torch.zeros(len(train_X), max_train_len, dtype=torch.bool)
        for i, x in enumerate(train_X):
            sl = x.shape[0]
            X_train_pad[i, :sl, :] = torch.from_numpy(x)
            mask_train[i, :sl] = True
        y_train_t = torch.tensor(train_y, dtype=torch.long)

        # Train probe
        probe = AttentionPoolingProbe(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            output_dim=NUM_CLASSES,
            max_seq_len=max_train_len,
        )
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        probe.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = probe(X_train_pad, mask_train)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()

        # Predict on held-out runs
        probe.eval()
        for idx in test_indices:
            x = X_list[idx]
            seq_len = x.shape[0]
            x_t = torch.from_numpy(x).float().unsqueeze(0)
            m = torch.ones(1, seq_len, dtype=torch.bool)
            with torch.no_grad():
                logits = probe(x_t, m)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
            all_preds[idx] = int(probs.argmax().item())
            all_probs[idx] = probs.numpy()

        if (fold_idx + 1) % 20 == 0:
            print(f"  Completed {fold_idx + 1}/{len(unique_anecdotes)} folds")

    return {
        "predictions": all_preds,
        "probabilities": all_probs,
        "true_labels": y,
        "anecdote_ids": anecdote_ids,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, per-class precision/recall, and confusion matrix."""
    n = len(y_true)
    accuracy = (y_true == y_pred).mean()

    # Weighted chance baseline
    class_counts = Counter(y_true.tolist())
    chance_baseline = max(class_counts.values()) / n

    # Confusion matrix: rows = true, cols = predicted
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": precision,
            "recall": recall,
            "support": int(conf_matrix[c, :].sum()),
        }

    return {
        "accuracy": float(accuracy),
        "chance_baseline": float(chance_baseline),
        "per_class": per_class,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_labels": [CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        "n_samples": n,
    }


def print_results(metrics: dict, label: str = "Overall"):
    """Pretty-print metrics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Accuracy:        {metrics['accuracy']:.3f}")
    print(f"  Chance baseline: {metrics['chance_baseline']:.3f}")
    print(f"  N samples:       {metrics['n_samples']}")

    print(f"\n  Per-class metrics:")
    for cls_name, vals in metrics["per_class"].items():
        print(
            f"    {cls_name:>15s}: "
            f"P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
            f"n={vals['support']}"
        )

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    labels = metrics["confusion_labels"]
    header = "            " + "  ".join(f"{l:>12s}" for l in labels)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>12d}" for v in row)
        print(f"    {labels[i]:>12s}  {row_str}")


def main():
    # ── Step 1: Instantiate tasks for both variants ───────────────────
    tasks = {}
    for variant in VARIANTS:
        tasks[variant] = ScruplesTask(
            subject_model=SUBJECT_MODEL,
            variant=variant,
            data_dir=DATA_DIR,
        )

    # ── Step 2: Extract activations if needed ─────────────────────────
    if EXTRACT_ACTIVATIONS:
        for variant in VARIANTS:
            task = tasks[variant]
            if not task.get_data():
                print(f"No data for variant '{variant}'. Run run_data() first.")
                return
            print(f"\nExtracting activations for variant '{variant}'...")
            task.extract_activations(
                model_name=ACTIVATION_MODEL,
                layers=[LAYER],
                load_in_4bit=LOAD_IN_4BIT,
                data_slice=DataSlice.all(),
            )

    # ── Step 3: Load 3-class dataset ──────────────────────────────────
    # Use the first task instance to call get_intervention_probe_data
    # (method reads CSVs directly by variant name, task instance just
    # provides data_dir)
    task = tasks[VARIANTS[0]]
    data_slice = DataSlice.all()

    print("\nLoading intervention probe data...")
    probe_data = task.get_intervention_probe_data(
        variants=VARIANTS,
        layer=LAYER,
        data_slice=data_slice,
    )

    X_list = probe_data["X_list"]
    y = probe_data["y"]
    anecdote_ids = probe_data["anecdote_ids"]
    run_ids = probe_data["run_ids"]

    print(f"Loaded {len(X_list)} samples")
    for c in range(NUM_CLASSES):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {(y == c).sum()} samples")
    print(f"  Unique anecdotes: {len(set(anecdote_ids))}")

    if len(X_list) < 10:
        print("Too few samples for cross-validation. Exiting.")
        return

    # ── Step 4: Leave-one-anecdote-out CV ─────────────────────────────
    print("\nRunning leave-one-anecdote-out cross-validation...")
    cv_results = leave_one_anecdote_out_cv(
        X_list=X_list,
        y=y,
        anecdote_ids=anecdote_ids,
    )

    # ── Step 5: Overall metrics ───────────────────────────────────────
    overall_metrics = compute_metrics(cv_results["true_labels"], cv_results["predictions"])
    print_results(overall_metrics, label="Overall 3-class intervention probe (LOAnecdoteO CV)")

    # ── Step 6: Strict sycophancy breakdown ───────────────────────────
    split = task.get_strict_sycophancy_split(variants=["suggest_wrong"])
    syc_set = set(split["syc_ids"])
    non_syc_set = set(split["non_syc_ids"])

    aid_arr = np.array(anecdote_ids)
    y_true = cv_results["true_labels"]
    y_pred = cv_results["predictions"]

    for subset_name, subset_ids in [("sycophantic", syc_set), ("non-sycophantic", non_syc_set)]:
        mask = np.array([a in subset_ids for a in aid_arr])
        if mask.sum() == 0:
            print(f"\n  No samples in {subset_name} subset, skipping.")
            continue
        subset_metrics = compute_metrics(y_true[mask], y_pred[mask])
        print_results(subset_metrics, label=f"Subset: {subset_name} anecdotes")

    # ── Step 7: Save results ──────────────────────────────────────────
    output_dir = DATA_DIR / "intervention_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "overall": overall_metrics,
        "config": {
            "variants": VARIANTS,
            "layer": LAYER,
            "activation_model": ACTIVATION_MODEL,
            "num_heads": NUM_HEADS,
            "lr": LR,
            "epochs": EPOCHS,
            "n_samples": len(X_list),
            "n_anecdotes": len(set(anecdote_ids)),
        },
    }

    for subset_name, subset_ids in [("sycophantic", syc_set), ("non-sycophantic", non_syc_set)]:
        mask = np.array([a in subset_ids for a in aid_arr])
        if mask.sum() > 0:
            results[subset_name] = compute_metrics(y_true[mask], y_pred[mask])

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
