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
from src2.tasks.scruples.prompts import INTERVENTION_SUGGESTED_ANSWER

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
LR = 8e-4  # Scaled up ~linearly with batch size (8→64)
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
EPOCHS = 60
BATCH_SIZE = 64
GRAD_CLIP = 1.0  # Gradient clipping
TEST_SPLIT = 0.2
VAL_SPLIT = 0.15  # fraction of train set for F1-based early stopping
PATIENCE = 15  # stop if val F1 doesn't improve for this many epochs
SWITCH_THRESHOLD = 0.40
HIGH_INTERVENTION_RATE = 0.82
LOW_INTERVENTION_RATE = 0.70
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


def _predict_labels(probe, X_list, device):
    """Get predicted class labels for a list of activations."""
    probe.eval()
    preds = []
    for x in X_list:
        seq_len = x.shape[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = probe(x_t, m)
            pred = logits.argmax(dim=-1).item()
        preds.append(pred)
    return np.array(preds)


def _compute_f1(y_true, y_pred):
    """Compute F1 for the sycophantic class (class 1)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def train_and_evaluate(
    train_X: list,
    train_y: np.ndarray,
    test_X: list,
    test_y: np.ndarray,
    val_X: list = None,
    val_y: np.ndarray = None,
    num_heads: int = NUM_HEADS,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    dropout: float = DROPOUT,
) -> dict:
    """Train probe with F1-based early stopping, evaluate on test set.

    Uses the provided validation set for early stopping. Selects
    the model checkpoint with the best validation F1 for the sycophantic class.
    """
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}, batch_size: {batch_size}")

    if val_X is None or val_y is None:
        raise ValueError("val_X and val_y must be provided")

    n_samples = len(train_X)
    print(f"  Train: {n_samples}, Val: {len(val_X)}, Test: {len(test_X)}")

    hidden_dim = train_X[0].shape[1]
    all_X = train_X + val_X + test_X
    max_seq_len = max(x.shape[0] for x in all_X) if all_X else 1

    class_counts = np.bincount(train_y, minlength=NUM_CLASSES)
    class_weights = n_samples / (NUM_CLASSES * class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"  Class counts: {class_counts.tolist()}, weights: {class_weights.tolist()}")

    probe = AttentionPoolingProbe(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        output_dim=NUM_CLASSES,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Track metrics for plotting
    history = {"epoch": [], "train_f1": [], "test_f1": []}

    # F1-based early stopping
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        probe.train()
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
            # Compute val F1 for early stopping every 5 epochs
            val_preds = _predict_labels(probe, val_X, device)
            val_f1 = _compute_f1(val_y, val_preds)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state = copy.deepcopy(probe.state_dict())
                no_improve = 0
            else:
                no_improve += 5  # we check every 5 epochs

            # Log train/test F1 for training curves (every 5 epochs)
            train_preds = _predict_labels(probe, train_X, device)
            test_preds_tmp = _predict_labels(probe, test_X, device)
            train_f1 = _compute_f1(train_y, train_preds)
            test_f1 = _compute_f1(test_y, test_preds_tmp)
            history["epoch"].append(epoch + 1)
            history["train_f1"].append(train_f1)
            history["test_f1"].append(test_f1)

            print(
                f"  Epoch {epoch + 1}/{epochs}, loss: {epoch_loss / n_batches:.4f}, "
                f"val_F1: {val_f1:.3f}, best: {best_val_f1:.3f} (ep {best_epoch}), "
                f"train_F1: {train_f1:.3f}, test_F1: {test_f1:.3f}"
            )

            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1} (no val F1 improvement for {PATIENCE} epochs)")
                break

    # Restore best model by val F1
    if best_state is not None:
        probe.load_state_dict(best_state)
        print(f"  Restored best model from epoch {best_epoch} (val_F1={best_val_f1:.3f})")

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
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
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
                data_slice=split_info,
            )

    # ── Step 4: Load sycophancy probe data ────────────────────────────
    print("\nLoading sycophancy probe data (uncertainty-robust split)...")
    probe_data = task.get_sycophancy_probe_data(
        variants=VARIANTS,
        layer=LAYER,
        data_slice=split_info,
        switch_threshold=SWITCH_THRESHOLD,
    )

    X_list_all = probe_data["X_list"]
    y_all = probe_data["y"]
    anecdote_ids_all = probe_data["anecdote_ids"]
    metadata_all = probe_data["metadata"]

    # ── Filter to intervention arm only (exclude control) ─────────────
    intv_mask = [m["arm"] == "intervention" for m in metadata_all]
    X_list_intv = [x for x, keep in zip(X_list_all, intv_mask) if keep]
    y_intv = y_all[np.array(intv_mask)]
    anecdote_ids_intv = [a for a, keep in zip(anecdote_ids_all, intv_mask) if keep]
    metadata_intv = [m for m, keep in zip(metadata_all, intv_mask) if keep]

    print(f"Loaded {len(X_list_all)} samples total, {len(X_list_intv)} intervention-only")

    # ── Filter to clean examples only ─────────────────────────────────
    # Keep: syc answer from syc prompts (label=1),
    #        control-majority answer from non-syc prompts (label=0)
    # Discard: non-syc runs from syc prompts, syc runs from non-syc prompts
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
            # Syc prompt: keep only if model gave sycophantic answer
            clean_mask.append(m["is_sycophantic_answer"])
        else:
            # Non-syc prompt: keep only if model gave majority control answer
            clean_mask.append(answer.upper() == majority_ctrl_answer)

    X_list = [x for x, keep in zip(X_list_intv, clean_mask) if keep]
    y = y_intv[np.array(clean_mask)]
    anecdote_ids = [a for a, keep in zip(anecdote_ids_intv, clean_mask) if keep]
    metadata = [m for m, keep in zip(metadata_intv, clean_mask) if keep]

    n_discarded = len(X_list_intv) - len(X_list)
    print(f"  Clean-example filter: kept {len(X_list)}, discarded {n_discarded}")
    print(f"  Class 0 (non_sycophantic): {(y == 0).sum()} samples")
    print(f"  Class 1 (sycophantic):     {(y == 1).sum()} samples")
    print(f"  Unique anecdotes: {len(set(anecdote_ids))}")

    # Print detailed statistics
    print_dataset_statistics(metadata, y)

    if len(X_list) < 10:
        print("Too few samples. Exiting.")
        return

    # ── Step 5: Train/val/test split (canonical from get_uncertainty_robust_split) ──
    train_aids = set(split_info.train_df["anecdote_id"].unique())
    val_aids = set(split_info.val_df["anecdote_id"].unique())
    test_aids = set(split_info.test_df["anecdote_id"].unique())

    train_idx = [i for i, a in enumerate(anecdote_ids) if a in train_aids]
    val_idx = [i for i, a in enumerate(anecdote_ids) if a in val_aids]
    test_idx = [i for i, a in enumerate(anecdote_ids) if a in test_aids]

    split = {
        "train_X": [X_list[i] for i in train_idx],
        "train_y": y[train_idx],
        "train_anecdote_ids": [anecdote_ids[i] for i in train_idx],
        "train_metadata": [metadata[i] for i in train_idx],
        "val_X": [X_list[i] for i in val_idx],
        "val_y": y[val_idx],
        "test_X": [X_list[i] for i in test_idx],
        "test_y": y[test_idx],
        "test_anecdote_ids": [anecdote_ids[i] for i in test_idx],
        "test_metadata": [metadata[i] for i in test_idx],
        "n_train_anecdotes": len(train_aids),
        "n_val_anecdotes": len(val_aids),
        "n_test_anecdotes": len(test_aids),
    }

    print(
        f"  Train: {len(split['train_X'])} samples from {split['n_train_anecdotes']} anecdotes"
    )
    print(
        f"  Val:   {len(split['val_X'])} samples from {split['n_val_anecdotes']} anecdotes"
    )
    print(
        f"  Test:  {len(split['test_X'])} samples from {split['n_test_anecdotes']} anecdotes"
    )

    # Print train/test statistics
    print("\n  Train set breakdown:")
    print_dataset_statistics(split["train_metadata"], split["train_y"])
    print("  Test set breakdown:")
    print_dataset_statistics(split["test_metadata"], split["test_y"])

    # ── Step 5: Parallel hyperparameter sweep ────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    SWEEP_CONFIGS = [
        {"weight_decay": 0.0,   "dropout": 0.0},
        {"weight_decay": 0.0,   "dropout": 0.1},
        {"weight_decay": 0.0,   "dropout": 0.3},
        {"weight_decay": 1e-4,  "dropout": 0.0},
        {"weight_decay": 1e-4,  "dropout": 0.1},
        {"weight_decay": 1e-4,  "dropout": 0.3},
        {"weight_decay": 1e-3,  "dropout": 0.0},
        {"weight_decay": 1e-3,  "dropout": 0.1},
        {"weight_decay": 1e-3,  "dropout": 0.3},
        {"weight_decay": 1e-2,  "dropout": 0.3},
    ]

    output_dir = DATA_DIR / "sycophancy_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _run_one(cfg):
        """Train a single probe config and return summary."""
        wd, do = cfg["weight_decay"], cfg["dropout"]
        results = train_and_evaluate(
            train_X=split["train_X"],
            train_y=split["train_y"],
            test_X=split["test_X"],
            test_y=split["test_y"],
            val_X=split["val_X"],
            val_y=split["val_y"],
            weight_decay=wd,
            dropout=do,
        )
        metrics = compute_metrics(results["true_labels"], results["predictions"])
        test_f1 = metrics["per_class"]["sycophantic"]["f1"]
        return {
            "weight_decay": wd,
            "dropout": do,
            "test_f1_syc": test_f1,
            "test_accuracy": metrics["accuracy"],
            "best_val_f1": results["best_val_f1"],
            "best_epoch": results["best_epoch"],
            "metrics": metrics,
            "results": results,
        }

    print(f"\nLaunching {len(SWEEP_CONFIGS)} probe configs in parallel...")
    sweep_results = []
    with ThreadPoolExecutor(max_workers=len(SWEEP_CONFIGS)) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in SWEEP_CONFIGS}
        for fut in as_completed(futures):
            r = fut.result()
            sweep_results.append(r)
            print(f"  Done: wd={r['weight_decay']:.0e} do={r['dropout']:.1f} → "
                  f"val_F1={r['best_val_f1']:.3f}  test_F1={r['test_f1_syc']:.3f}  "
                  f"acc={r['test_accuracy']:.3f}  ep={r['best_epoch']}")

    # Sort by test F1 descending
    sweep_results.sort(key=lambda r: r["test_f1_syc"], reverse=True)
    best = sweep_results[0]

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SWEEP SUMMARY (sorted by test F1)")
    print(f"{'='*60}")
    print(f"  {'weight_decay':>12s}  {'dropout':>7s}  {'val_F1':>6s}  {'test_F1':>7s}  {'acc':>5s}  {'ep':>3s}")
    for r in sweep_results:
        marker = " <-- best" if r is best else ""
        print(f"  {r['weight_decay']:12.1e}  {r['dropout']:7.1f}  {r['best_val_f1']:6.3f}  {r['test_f1_syc']:7.3f}  {r['test_accuracy']:5.3f}  {r['best_epoch']:3d}{marker}")

    print(f"\n  Best config: weight_decay={best['weight_decay']}, dropout={best['dropout']}")
    print(f"  Best test F1 (sycophantic): {best['test_f1_syc']:.3f}")

    # Print full metrics for the best
    print_results(best["metrics"], label="Best Config Results")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "sweep_results": [
            {k: v for k, v in r.items() if k not in ("metrics", "results")}
            for r in sweep_results
        ],
        "best_config": {"weight_decay": best["weight_decay"], "dropout": best["dropout"]},
        "best_test_metrics": best["metrics"],
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
            "n_val_samples": len(split["val_X"]),
            "n_test_samples": len(split["test_X"]),
            "n_train_anecdotes": split["n_train_anecdotes"],
            "n_val_anecdotes": split["n_val_anecdotes"],
            "n_test_anecdotes": split["n_test_anecdotes"],
        },
    }

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSweep results saved to {output_dir / 'sweep_results.json'}")

    # Save and plot training curves for the best config
    best_history = best["results"].get("history", {})
    if best_history and best_history.get("epoch"):
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(best_history, f, indent=2)
        print(f"Training history saved to {output_dir / 'training_history.json'}")

        from src2.utils.plotting import plot_training_curves
        plot_training_curves(
            best_history,
            metric_name="f1",
            output_path=output_dir / "training_curves.png",
            title=f"Best Probe (wd={best['weight_decay']}, do={best['dropout']}): F1 vs Epoch",
        )


if __name__ == "__main__":
    main()
