#!/usr/bin/env python3
"""
Sycophancy attention probe.

Trains a binary attention probe that predicts whether a run is sycophantic
(model switched to agree with user suggestion) from CoT + response activations.

Label = 1: intervention run where model switched (is_sycophantic=True)
Label = 0: control run, or intervention run that didn't switch

Usage:
    python -m src2.runs.run_sycophancy_probe
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src2.data_slice import DataSlice
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
CLASS_NAMES = {0: "non_sycophantic", 1: "sycophantic"}
NUM_CLASSES = 2

# Training hyperparameters
NUM_HEADS = 4
LR = 1e-4  # Lower LR for stability
EPOCHS = 40
BATCH_SIZE = 8
GRAD_CLIP = 1.0  # Gradient clipping
TEST_SPLIT = 0.2
SWITCH_THRESHOLD = 0.40
HIGH_INTERVENTION_RATE = 0.82
LOW_INTERVENTION_RATE = 0.60
N_SYC_HIGH_PER_VARIANT = 25
N_SYC_LOW_PER_VARIANT = 25
N_NON_SYC_PER_VARIANT = 50
SEED = 42

EXTRACT_ACTIVATIONS = False


def print_dataset_statistics(metadata: list, y: np.ndarray):
    """Print breakdown of dataset by variant/arm/answer."""
    print("\n" + "=" * 70)
    print("  Dataset Statistics")
    print("=" * 70)

    # Count by group
    groups = {}
    for i, m in enumerate(metadata):
        arm = m["arm"]
        variant = m["variant"] if arm == "intervention" else "control"
        answer = m["answer"]
        key = (variant, answer)
        if key not in groups:
            groups[key] = {
                "count": 0,
                "syco_answer": 0,  # answer matches suggestion
                "prompt_syco": 0,  # switch_rate > 0.5
                "label": 0,  # both conditions met
            }
        groups[key]["count"] += 1
        if arm == "intervention":
            groups[key]["syco_answer"] += int(m.get("is_sycophantic_answer", False))
            groups[key]["prompt_syco"] += int(m.get("prompt_is_sycophantic", False))
        groups[key]["label"] += int(y[i])

    total = len(metadata)

    # Print in organized order
    order = [
        ("control", "A"), ("control", "B"),
        ("suggest_wrong", "A"), ("suggest_wrong", "B"),
        ("suggest_right", "A"), ("suggest_right", "B"),
    ]

    print(f"\n  {'Group':<20} {'N':>6} {'SycoAns':>8} {'PromptSyco':>11} {'Label=1':>8}")
    print("  " + "-" * 53)

    for variant, answer in order:
        key = (variant, answer)
        if key in groups:
            g = groups[key]
            n = g["count"]
            sa = g["syco_answer"]
            ps = g["prompt_syco"]
            lb = g["label"]
            sa_pct = f"({100*sa/n:.0f}%)" if n > 0 else ""
            ps_pct = f"({100*ps/n:.0f}%)" if n > 0 else ""
            lb_pct = f"({100*lb/n:.0f}%)" if n > 0 else ""
            print(f"  {variant + ' ' + answer:<20} {n:>6} {sa:>4}{sa_pct:>4} {ps:>5}{ps_pct:>6} {lb:>4}{lb_pct:>4}")
        else:
            print(f"  {variant + ' ' + answer:<20} {0:>6} {0:>8} {0:>11} {0:>8}")

    print("  " + "-" * 53)
    print(f"  {'TOTAL':<20} {total:>6} {'-':>8} {'-':>11} {int(y.sum()):>4}({100*y.mean():.0f}%)")
    print()
    print("  SycoAns = answer matches sycophantic suggestion")
    print("  PromptSyco = anecdote switch_rate > 0.5")
    print("  Label=1 = SycoAns AND PromptSyco (for intervention runs)")
    print()


def train_test_split_by_anecdote(
    X_list: list,
    y: np.ndarray,
    anecdote_ids: list,
    metadata: list,
    test_fraction: float = TEST_SPLIT,
    seed: int = SEED,
    anecdote_strata: dict = None,
) -> dict:
    """Split data into train/test by anecdote, with optional stratification.

    When *anecdote_strata* is provided (mapping anecdote_id -> stratum label),
    each stratum is split independently so that the stratum proportions are
    preserved in both train and test sets.
    """
    rng = np.random.default_rng(seed)

    if anecdote_strata is not None:
        # Group anecdotes by stratum
        strata_groups: dict = {}
        for aid in set(anecdote_ids):
            s = anecdote_strata.get(aid, "unknown")
            strata_groups.setdefault(s, []).append(aid)

        train_anecdotes: set = set()
        test_anecdotes: set = set()

        for stratum, aids in sorted(strata_groups.items()):
            aids = sorted(aids)
            rng.shuffle(aids)
            n_test = max(1, int(len(aids) * test_fraction))
            test_anecdotes.update(aids[:n_test])
            train_anecdotes.update(aids[n_test:])

        # Print stratification summary
        print("  Stratified split by anecdote:")
        for stratum in sorted(strata_groups):
            aids = strata_groups[stratum]
            n_train = sum(1 for a in aids if a in train_anecdotes)
            n_test = sum(1 for a in aids if a in test_anecdotes)
            print(f"    {stratum}: {n_train} train, {n_test} test")
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
        "train_metadata": [metadata[i] for i in train_idx],
        "test_X": [X_list[i] for i in test_idx],
        "test_y": y[test_idx],
        "test_anecdote_ids": [anecdote_ids[i] for i in test_idx],
        "test_metadata": [metadata[i] for i in test_idx],
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
    max_seq_len = max(
        max(x.shape[0] for x in train_X),
        max(x.shape[0] for x in test_X),
    )
    n_samples = len(train_X)

    # Compute class weights (inverse frequency)
    class_counts = np.bincount(train_y, minlength=NUM_CLASSES)
    class_weights = n_samples / (NUM_CLASSES * class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"  Class counts: {class_counts.tolist()}, weights: {class_weights.tolist()}")

    probe = AttentionPoolingProbe(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        output_dim=NUM_CLASSES,
        max_seq_len=max_seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    probe.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
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

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / n_batches:.4f}")

    # Evaluate
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
    """Compute binary classification metrics."""
    n = len(y_true)
    accuracy = (y_true == y_pred).mean()

    class_counts = Counter(y_true.tolist())
    chance_baseline = max(class_counts.values()) / n

    # Confusion matrix
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    # Per-class metrics
    per_class = {}
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
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
    print("=" * 60)
    print(f"  Accuracy:        {metrics['accuracy']:.3f}")
    print(f"  Chance baseline: {metrics['chance_baseline']:.3f}")
    print(f"  N samples:       {metrics['n_samples']}")

    print(f"\n  Per-class metrics:")
    for cls_name, vals in metrics["per_class"].items():
        print(
            f"    {cls_name:>15s}: "
            f"P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
            f"F1={vals['f1']:.3f}  n={vals['support']}"
        )

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    labels = metrics["confusion_labels"]
    header = "            " + "  ".join(f"{l:>15s}" for l in labels)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>15d}" for v in row)
        print(f"    {labels[i]:>15s}  {row_str}")


def main():
    # ── Step 1: Instantiate tasks ─────────────────────────────────────
    tasks = {}
    for variant in VARIANTS:
        tasks[variant] = ScruplesTask(
            subject_model=SUBJECT_MODEL,
            variant=variant,
            data_dir=DATA_DIR,
        )

    # ── Step 2: Compute uncertainty-robust split ────────────────────────
    task = tasks[VARIANTS[0]]
    print("\nComputing uncertainty-robust split...")
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

    # ── Step 3: Extract activations if needed (only for split) ────────
    if EXTRACT_ACTIVATIONS:
        for variant in VARIANTS:
            t = tasks[variant]
            if not t.get_data():
                print(f"No data for variant '{variant}'. Run run_data() first.")
                return
            print(f"\nExtracting activations for variant '{variant}'...")
            t.extract_activations(
                model_name=ACTIVATION_MODEL,
                layers=[LAYER],
                load_in_4bit=LOAD_IN_4BIT,
                data_slice=split_info["data_slice"],
            )

    # ── Step 4: Load sycophancy probe data ────────────────────────────
    print("\nLoading sycophancy probe data (uncertainty-robust split)...")
    probe_data = task.get_sycophancy_probe_data(
        variants=VARIANTS,
        layer=LAYER,
        data_slice=split_info["data_slice"],
        switch_threshold=SWITCH_THRESHOLD,
    )

    X_list = probe_data["X_list"]
    y = probe_data["y"]
    anecdote_ids = probe_data["anecdote_ids"]
    metadata = probe_data["metadata"]

    print(f"Loaded {len(X_list)} samples")
    print(f"  Class 0 (non_sycophantic): {(y == 0).sum()} samples")
    print(f"  Class 1 (sycophantic):     {(y == 1).sum()} samples")
    print(f"  Unique anecdotes: {len(set(anecdote_ids))}")

    # Print detailed statistics
    print_dataset_statistics(metadata, y)

    if len(X_list) < 10:
        print("Too few samples. Exiting.")
        return

    # ── Step 5: Train/test split (stratified by uncertainty strata) ──
    print(f"Splitting data ({1 - TEST_SPLIT:.0%} train, {TEST_SPLIT:.0%} test)...")
    split = train_test_split_by_anecdote(
        X_list, y, anecdote_ids, metadata,
        anecdote_strata=split_info.get("anecdote_strata"),
    )

    print(
        f"  Train: {len(split['train_X'])} samples from {split['n_train_anecdotes']} anecdotes"
    )
    print(
        f"  Test:  {len(split['test_X'])} samples from {split['n_test_anecdotes']} anecdotes"
    )

    # Print train/test statistics
    print("\n  Train set breakdown:")
    print_dataset_statistics(split["train_metadata"], split["train_y"])
    print("  Test set breakdown:")
    print_dataset_statistics(split["test_metadata"], split["test_y"])

    # ── Step 5: Train and evaluate ────────────────────────────────────
    print("Training sycophancy attention probe...")
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
    output_dir = DATA_DIR / "sycophancy_probe"
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
            "batch_size": BATCH_SIZE,
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
