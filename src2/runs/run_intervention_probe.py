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

from src2.data_slice import DataSlice
from src2.methods import AttentionProbe
from src2.methods.attention_probe import AttentionPoolingProbe
from src2.tasks import ScruplesTask

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
BATCH_SIZE = 8  # Reduce if OOM, increase for faster training
TEST_SPLIT = 0.2
SEED = 42

# if false, training automatically uses whatever activations we have.
EXTRACT_ACTIVATIONS = False


def train_test_split_by_anecdote(
    X_list: list,
    y: np.ndarray,
    anecdote_ids: list,
    test_fraction: float = TEST_SPLIT,
    seed: int = SEED,
) -> dict:
    """
    Split data into train/test by anecdote (all runs from an anecdote go together).
    """
    rng = np.random.default_rng(seed)
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
        "n_train_anecdotes": len(train_anecdotes),
        "n_test_anecdotes": len(test_anecdotes),
    }


def train_and_evaluate(
    train_X: list,
    train_y: np.ndarray,
    test_X: list,
    test_y: np.ndarray,
    num_heads: int = NUM_HEADS,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train probe on train set, evaluate on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}, batch_size: {batch_size}")

    hidden_dim = train_X[0].shape[1]
    max_train_len = max(x.shape[0] for x in train_X)
    n_samples = len(train_X)

    # Train probe
    probe = AttentionPoolingProbe(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        output_dim=NUM_CLASSES,
        max_seq_len=max_train_len,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(epochs):
        # Shuffle indices each epoch
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]

            # Pad batch (only to max length in this batch)
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
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / n_batches:.4f}")

    # Evaluate on test set
    probe.eval()
    test_preds = []
    test_probs = []
    for x in test_X:
        seq_len = x.shape[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = probe(x_t, m)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        test_preds.append(int(probs.argmax().item()))
        test_probs.append(probs.cpu().numpy())

    return {
        "predictions": np.array(test_preds),
        "probabilities": np.array(test_probs),
        "true_labels": test_y,
        "final_train_loss": float(epoch_loss / n_batches),
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


def print_results(metrics: dict, label: str = "Test Set"):
    """Pretty-print metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
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
    task = tasks[VARIANTS[0]]

    print("\nLoading intervention probe data...")
    probe_data = task.get_intervention_probe_data(
        variants=VARIANTS,
        layer=LAYER,
        data_slice=DataSlice.all(),
    )

    X_list = probe_data["X_list"]
    y = probe_data["y"]
    anecdote_ids = probe_data["anecdote_ids"]

    print(f"Loaded {len(X_list)} samples")
    for c in range(NUM_CLASSES):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {(y == c).sum()} samples")
    print(f"  Unique anecdotes: {len(set(anecdote_ids))}")

    if len(X_list) < 10:
        print("Too few samples. Exiting.")
        return

    # ── Step 4: Train/test split by anecdote ──────────────────────────
    print(f"\nSplitting data ({1 - TEST_SPLIT:.0%} train, {TEST_SPLIT:.0%} test)...")
    split = train_test_split_by_anecdote(X_list, y, anecdote_ids)

    print(
        f"  Train: {len(split['train_X'])} samples from {split['n_train_anecdotes']} anecdotes"
    )
    print(
        f"  Test:  {len(split['test_X'])} samples from {split['n_test_anecdotes']} anecdotes"
    )

    # ── Step 5: Train and evaluate ────────────────────────────────────
    print("\nTraining attention probe...")
    results = train_and_evaluate(
        train_X=split["train_X"],
        train_y=split["train_y"],
        test_X=split["test_X"],
        test_y=split["test_y"],
    )

    # ── Step 6: Compute and print metrics ─────────────────────────────
    metrics = compute_metrics(results["true_labels"], results["predictions"])
    print_results(metrics, label="Test Set Results")

    # ── Step 7: Save results ──────────────────────────────────────────
    output_dir = DATA_DIR / "intervention_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "test_metrics": metrics,
        "config": {
            "variants": VARIANTS,
            "layer": LAYER,
            "activation_model": ACTIVATION_MODEL,
            "num_heads": NUM_HEADS,
            "lr": LR,
            "epochs": EPOCHS,
            "test_split": TEST_SPLIT,
            "seed": SEED,
            "n_train_samples": len(split["train_X"]),
            "n_test_samples": len(split["test_X"]),
            "n_train_anecdotes": split["n_train_anecdotes"],
            "n_test_anecdotes": split["n_test_anecdotes"],
        },
        "final_train_loss": results["final_train_loss"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
