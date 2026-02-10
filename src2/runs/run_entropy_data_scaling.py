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
import torch.nn.functional as F

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


def precompute_pooled_features(X_list, device, batch_size=128):
    """Mean-pool variable-length sequences into a dense [N, D] tensor on GPU.

    This is the key optimisation: mean-pooling has no learnable parameters, so
    we do the expensive pad-and-transfer step exactly once for the entire
    dataset instead of once per batch per epoch per probe.
    """
    hidden_dim = X_list[0].shape[1]
    N = len(X_list)
    pooled = torch.zeros(N, hidden_dim, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = X_list[start:end]
        max_len = max(x.shape[0] for x in batch)

        X_pad = torch.zeros(len(batch), max_len, hidden_dim, device=device)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)
        for i, x in enumerate(batch):
            sl = x.shape[0]
            X_pad[i, :sl] = torch.from_numpy(x).to(device)
            mask[i, :sl] = True

        lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled[start:end] = (X_pad * mask.unsqueeze(-1).float()).sum(dim=1) / lengths

    return pooled


def train_all_probes_batched(train_pooled, train_y, test_pooled, test_y,
                              data_fractions, n_repeats, seed):
    """Train all data-scaling probes *simultaneously*.

    All probes are stacked into a single weight matrix [n_probes, D] so each
    training step is one batched matmul. A boolean membership mask assigns
    different data subsets to different probes, and per-probe early stopping
    freezes converged probes while the rest keep training.
    """
    device = train_pooled.device
    D = train_pooled.shape[1]
    N_train = train_pooled.shape[0]
    n_probes = len(data_fractions) * n_repeats

    # Targets as GPU tensors
    train_y_t = (torch.from_numpy(train_y).float().to(device)
                 if isinstance(train_y, np.ndarray) else train_y.float().to(device))
    test_y_np = test_y if isinstance(test_y, np.ndarray) else test_y.cpu().numpy()

    # ── Build membership masks: [n_probes, N_train] ──────────────────
    membership = torch.zeros(n_probes, N_train, dtype=torch.bool, device=device)
    probe_idx = 0
    for frac in data_fractions:
        n_subset = max(5, int(N_train * frac))
        for rep in range(n_repeats):
            rng = np.random.default_rng(seed + rep)
            indices = rng.choice(N_train, n_subset, replace=False)
            membership[probe_idx, indices] = True
            probe_idx += 1

    # ── Initialise all probes as a single weight matrix ──────────────
    W = nn.Parameter(torch.empty(n_probes, D, device=device))
    b = nn.Parameter(torch.zeros(n_probes, device=device))
    # Match nn.Linear default init (kaiming_uniform with a=sqrt(5))
    nn.init.kaiming_uniform_(W, a=5 ** 0.5)

    optimizer = torch.optim.AdamW([W, b], lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    # Per-probe early stopping state
    best_loss = torch.full((n_probes,), float("inf"))
    best_W = W.data.clone()
    best_b = b.data.clone()
    no_improve = torch.zeros(n_probes, dtype=torch.long)
    active = torch.ones(n_probes, dtype=torch.bool)        # CPU
    active_dev = active.to(device)

    for epoch in range(EPOCHS):
        perm = torch.randperm(N_train, device=device)
        epoch_loss_sum = torch.zeros(n_probes, device=device)
        epoch_count = torch.zeros(n_probes, device=device)

        for start in range(0, N_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_train)
            idx = perm[start:end]

            x_batch = train_pooled[idx]             # [B, D]
            y_batch = train_y_t[idx]                # [B]

            # Dropout (shared mask across probes – fine for linear probes)
            x_drop = F.dropout(x_batch, p=DROPOUT, training=True)

            # Forward: [B, D] @ [D, n_probes] + [n_probes] → [B, n_probes]
            preds = x_drop @ W.T + b

            # Per-element squared error
            sq_err = (preds - y_batch.unsqueeze(1)) ** 2   # [B, n_probes]

            # Membership + active mask
            batch_member = membership[:, idx].T.float()     # [B, n_probes]
            batch_mask = batch_member * active_dev.unsqueeze(0).float()

            counts = batch_mask.sum(dim=0).clamp(min=1)
            per_probe_loss = (sq_err * batch_mask).sum(dim=0) / counts
            loss = per_probe_loss.sum()

            optimizer.zero_grad()
            loss.backward()

            # Zero grads for frozen probes (prevent weight-decay drift)
            with torch.no_grad():
                inactive = ~active_dev
                if inactive.any():
                    W.grad[inactive] = 0
                    b.grad[inactive] = 0

            torch.nn.utils.clip_grad_norm_([W, b], GRAD_CLIP)
            optimizer.step()

            # Restore frozen probes (counteract any residual optimizer state)
            with torch.no_grad():
                if inactive.any():
                    W.data[inactive] = best_W[inactive].to(device)
                    b.data[inactive] = best_b[inactive].to(device)

            # Accumulate epoch loss for early stopping
            with torch.no_grad():
                epoch_loss_sum += (sq_err.detach() * batch_member).sum(dim=0)
                epoch_count += batch_member.sum(dim=0)

        scheduler.step()

        # ── Per-probe early stopping ──────────────────────────────────
        avg_loss = (epoch_loss_sum / epoch_count.clamp(min=1)).cpu()
        for j in range(n_probes):
            if not active[j]:
                continue
            if avg_loss[j].item() < best_loss[j].item() - MIN_DELTA:
                best_loss[j] = avg_loss[j]
                best_W[j] = W.data[j].clone()
                best_b[j] = b.data[j].clone()
                no_improve[j] = 0
            else:
                no_improve[j] += 1
            if no_improve[j] >= PATIENCE:
                active[j] = False
                W.data[j] = best_W[j].to(device)
                b.data[j] = best_b[j].to(device)

        active_dev = active.to(device)
        if not active.any():
            break

    # ── Restore all best weights & evaluate ──────────────────────────
    W.data.copy_(best_W.to(device))
    b.data.copy_(best_b.to(device))

    with torch.no_grad():
        test_preds = (test_pooled @ W.T + b).cpu().numpy()  # [N_test, n_probes]

    ss_tot = np.sum((test_y_np - test_y_np.mean()) ** 2)
    r2_list = []
    for j in range(n_probes):
        ss_res = np.sum((test_y_np - test_preds[:, j]) ** 2)
        r2_list.append(float(1 - ss_res / max(ss_tot, 1e-8)))

    return r2_list


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

    # ── Pre-compute mean-pooled features (once) ──────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPre-computing pooled features on {device}...")
    train_pooled = precompute_pooled_features(full_train_X, device)
    test_pooled = precompute_pooled_features(test_X, device)
    print(f"  Train: {train_pooled.shape}, Test: {test_pooled.shape}")

    # ── Train all probes simultaneously ──────────────────────────────
    n_probes = len(DATA_FRACTIONS) * N_REPEATS
    print(f"\nTraining {n_probes} probes simultaneously ({N_REPEATS} repeats × {len(DATA_FRACTIONS)} fractions)...")
    r2_all = train_all_probes_batched(
        train_pooled, full_train_y, test_pooled, test_y,
        DATA_FRACTIONS, N_REPEATS, SEED,
    )

    # ── Reshape results ──────────────────────────────────────────────
    sizes = []
    mean_r2s = []
    std_r2s = []

    for i, frac in enumerate(DATA_FRACTIONS):
        n_subset = max(5, int(n_train * frac))
        r2_scores = r2_all[i * N_REPEATS : (i + 1) * N_REPEATS]
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
