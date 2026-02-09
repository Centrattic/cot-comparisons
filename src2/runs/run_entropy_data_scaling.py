#!/usr/bin/env python3
"""
Data scaling analysis for entropy probe.

Trains the entropy probe (mean-pool linear or attention) with increasing
fractions of training data and evaluates test R² at each size. Produces a
plot of test R² vs training set size to check whether more data would help.

Usage:
    python -m src2.runs.run_entropy_data_scaling
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src2.data_slice import DataSlice
from src2.tasks.forced_response.task import ForcingTask
from src2.utils.questions import load_gpqa_from_huggingface

# ── Import helpers from run_entropy_probe ────────────────────────────
from src2.runs.run_entropy_probe import (
    MeanPoolLinearProbe,
    build_question_splits,
    dist_dict_to_entropy,
    load_probe_data,
    mean_subtract_per_question,
    sample_sentence_indices,
)

# ── Configuration (mirrors run_entropy_probe.py) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

MAX_SENTENCES_PER_QUESTION_TRAIN = 50
MAX_SENTENCES_PER_QUESTION_EVAL = 50

# Training hyperparameters
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 500
BATCH_SIZE = 256
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.05
PATIENCE = 50
MIN_DELTA = 0.001
MEAN_SUBTRACT = True
TRIM_TO_COT = True
SEED = 42

# Data scaling fractions
DATA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 3


# ── Training helper ──────────────────────────────────────────────────


def _eval_r2(probe, X_list, y, device, batch_size=64):
    """Compute R² on a dataset."""
    probe.eval()
    hidden_dim = X_list[0].shape[1]
    all_preds = []

    for start in range(0, len(X_list), batch_size):
        end = min(start + batch_size, len(X_list))
        batch_X = X_list[start:end]
        batch_max_len = max(x.shape[0] for x in batch_X)

        X_pad = torch.zeros(len(batch_X), batch_max_len, hidden_dim, device=device)
        mask = torch.zeros(len(batch_X), batch_max_len, dtype=torch.bool, device=device)
        for i, x in enumerate(batch_X):
            sl = x.shape[0]
            X_pad[i, :sl, :] = torch.from_numpy(x).to(device)
            mask[i, :sl] = True

        with torch.no_grad():
            pred = probe(X_pad, mask)
        all_preds.append(pred.cpu().numpy())

    preds = np.concatenate(all_preds)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-8))


def train_and_get_r2(train_X, train_y, test_X, test_y):
    """Train entropy probe on given data and return test R²."""
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = train_X[0].shape[1]
    n_samples = len(train_X)

    probe = MeanPoolLinearProbe(input_dim=hidden_dim, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        probe.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

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
            y_t = torch.from_numpy(batch_y).float().to(device)

            optimizer.zero_grad()
            pred = probe(X_pad, mask)
            loss = loss_fn(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Simple early stopping on training loss (no separate val split here)
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_loss
            best_state = copy.deepcopy(probe.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)

    return _eval_r2(probe, test_X, test_y, device)


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # ── Build question splits ─────────────────────────────────────────
    print("Building question splits...")
    splits = build_question_splits(forcing.forcing_dir)
    train_ids = splits["train_ids"]
    val_ids = splits["val_ids"]
    eval_ids = splits["eval_ids"]

    all_ids = train_ids + val_ids + eval_ids

    # ── Sample sentences ──────────────────────────────────────────────
    print("Sampling sentence indices...")
    train_smap = sample_sentence_indices(
        forcing.forcing_dir, train_ids, MAX_SENTENCES_PER_QUESTION_TRAIN, SEED,
    )
    val_smap = sample_sentence_indices(
        forcing.forcing_dir, val_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    eval_smap = sample_sentence_indices(
        forcing.forcing_dir, eval_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    smap = {**train_smap, **val_smap, **eval_smap}

    # ── Load data ─────────────────────────────────────────────────────
    tokenizer = None
    if TRIM_TO_COT:
        from transformers import AutoTokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(ACTIVATION_MODEL, trust_remote_code=True)

    # Use train + val as the full training pool, eval as fixed test set
    full_train_ids = train_ids + val_ids

    print("Loading training data...")
    train_data = load_probe_data(
        forcing, full_train_ids, smap, LAYER, "full_sequence",
        tokenizer=tokenizer, trim_to_cot=TRIM_TO_COT,
    )
    print(f"  Train pool: {len(train_data['X_list'])} samples")

    print("Loading eval data...")
    eval_data = load_probe_data(
        forcing, eval_ids, smap, LAYER, "full_sequence",
        tokenizer=tokenizer, trim_to_cot=TRIM_TO_COT,
    )
    print(f"  Eval: {len(eval_data['X_list'])} samples")

    if len(train_data["X_list"]) < 10 or len(eval_data["X_list"]) < 5:
        print("Too few samples. Exiting.")
        return

    # ── Mean subtraction ──────────────────────────────────────────────
    if MEAN_SUBTRACT:
        print("Applying per-question mean subtraction...")
        train_data = mean_subtract_per_question(train_data)
        eval_data = mean_subtract_per_question(eval_data)

    full_train_X = train_data["X_list"]
    full_train_y = train_data["y_entropy"]
    test_X = eval_data["X_list"]
    test_y = eval_data["y_entropy"]
    n_train = len(full_train_X)

    # ── Data scaling ──────────────────────────────────────────────────
    print(f"\nRunning data scaling analysis ({N_REPEATS} repeats per fraction)...")
    sizes = []
    mean_r2s = []
    std_r2s = []

    for frac in DATA_FRACTIONS:
        n_subset = max(5, int(n_train * frac))
        r2_scores = []

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(SEED + rep)
            indices = rng.choice(n_train, n_subset, replace=False)
            sub_X = [full_train_X[i] for i in indices]
            sub_y = full_train_y[indices]

            r2 = train_and_get_r2(sub_X, sub_y, test_X, test_y)
            r2_scores.append(r2)

        mean_r2 = float(np.mean(r2_scores))
        std_r2 = float(np.std(r2_scores))
        sizes.append(n_subset)
        mean_r2s.append(mean_r2)
        std_r2s.append(std_r2)
        print(f"  {frac*100:5.1f}% ({n_subset:4d} samples): R² = {mean_r2:.3f} ± {std_r2:.3f}")

    # ── Save results ──────────────────────────────────────────────────
    output_dir = DATA_DIR / "data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "fractions": DATA_FRACTIONS,
        "sizes": sizes,
        "mean_r2": mean_r2s,
        "std_r2": std_r2s,
        "n_repeats": N_REPEATS,
        "n_train_full": n_train,
        "n_test": len(test_X),
    }
    with open(output_dir / "entropy_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    from src2.utils.plotting import plot_data_scaling
    plot_data_scaling(
        sizes=sizes,
        scores=mean_r2s,
        metric_name="r2",
        output_path=output_dir / "entropy_data_scaling.png",
        title="Entropy Probe: Test R² vs Training Data Size",
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
