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

# Training hyperparameters  (must match run_entropy_probe.py)
BOTTLENECK_DIM = 8
NUM_HEADS = 1
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 500
BATCH_SIZE = 256
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.05
PATIENCE = 50
MIN_DELTA = 0.001
MEAN_SUBTRACT = False
TRIM_TO_COT = True
SEED = 42

# Data scaling fractions
DATA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 3


# ── Helpers ──────────────────────────────────────────────────────────


def _pad_batch(X_list, indices, device):
    """Pad a batch of variable-length sequences and move to GPU."""
    batch = [X_list[i] for i in indices]
    hidden_dim = batch[0].shape[1]
    max_len = max(x.shape[0] for x in batch)
    B = len(batch)
    X_pad = torch.zeros(B, max_len, hidden_dim, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for i, x in enumerate(batch):
        sl = x.shape[0]
        X_pad[i, :sl] = torch.from_numpy(x).to(device)
        mask[i, :sl] = True
    return X_pad, mask


# ── Batched bottleneck attention probes ──────────────────────────────


class BatchedBottleneckAttentionProbes(nn.Module):
    """P BottleneckAttentionProbes trained simultaneously via batched einsum.

    Each probe is: Linear(D, bottleneck) → LayerNorm → GELU → Dropout
                   → AttentionPooling(bottleneck, num_heads) → scalar

    All weights are stacked along a leading *probe* dimension.  The attention
    part uses the pool-first-then-project trick (value projection is linear,
    so sum(attn * (x @ W)) == (sum(attn * x)) @ W) to avoid materialising
    a large ``[B, P, S, V]`` tensor.
    """

    def __init__(self, n_probes, input_dim, bottleneck_dim, num_heads,
                 max_seq_len, dropout):
        super().__init__()
        self.n_probes = n_probes
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.head_dim = bottleneck_dim // num_heads
        self.eps = 1e-5  # LayerNorm epsilon

        # ── Bottleneck projection: Linear(D, bd) + LayerNorm(bd) ─────
        self.proj_w = nn.Parameter(torch.empty(n_probes, bottleneck_dim, input_dim))
        self.proj_b = nn.Parameter(torch.zeros(n_probes, bottleneck_dim))
        self.ln_w = nn.Parameter(torch.ones(n_probes, bottleneck_dim))
        self.ln_b = nn.Parameter(torch.zeros(n_probes, bottleneck_dim))

        # ── Attention: query_proj, position_bias, value_proj ─────────
        self.query_w = nn.Parameter(torch.empty(n_probes, num_heads, bottleneck_dim))
        self.query_b = nn.Parameter(torch.zeros(n_probes, num_heads))
        self.pos_bias = nn.Parameter(torch.zeros(n_probes, num_heads, max_seq_len))
        self.value_w = nn.Parameter(
            torch.empty(n_probes, num_heads, self.head_dim, bottleneck_dim))
        self.value_b = nn.Parameter(
            torch.zeros(n_probes, num_heads, self.head_dim))

        # ── Classifier: dropout → Linear(bd, 1) → scalar ────────────
        self.class_w = nn.Parameter(
            torch.empty(n_probes, 1, num_heads * self.head_dim))
        self.class_b = nn.Parameter(torch.zeros(n_probes, 1))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Match nn.Linear / nn.LayerNorm defaults per probe."""
        for p in range(self.n_probes):
            # Bottleneck projection
            nn.init.kaiming_uniform_(self.proj_w.data[p], a=5 ** 0.5)
            bound = 1.0 / self.input_dim ** 0.5
            nn.init.uniform_(self.proj_b.data[p], -bound, bound)
            # ln_w=1, ln_b=0 already set by default

            # Query projection
            nn.init.kaiming_uniform_(self.query_w.data[p], a=5 ** 0.5)
            bound_q = 1.0 / self.bottleneck_dim ** 0.5
            nn.init.uniform_(self.query_b.data[p], -bound_q, bound_q)

            # Value projection
            w_flat = self.value_w.data[p].view(-1, self.bottleneck_dim)
            nn.init.kaiming_uniform_(w_flat, a=5 ** 0.5)
            nn.init.uniform_(self.value_b.data[p].view(-1), -bound_q, bound_q)

            # Classifier
            nn.init.kaiming_uniform_(self.class_w.data[p], a=5 ** 0.5)
            fan_in = self.num_heads * self.head_dim
            nn.init.uniform_(self.class_b.data[p], -1.0 / fan_in ** 0.5,
                             1.0 / fan_in ** 0.5)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, S, D]
            mask: [B, S] boolean
        Returns:
            [B, P]  (scalar entropy prediction per probe)
        """
        B, S, D = x.shape
        P = self.n_probes
        H = self.num_heads
        hd = self.head_dim
        bd = self.bottleneck_dim

        # ── 1) Bottleneck projection: [B, S, D] → [B, P, S, bd] ─────
        h = (torch.einsum('bsd,pnd->bpsn', x, self.proj_w)
             + self.proj_b[None, :, None, :])

        # Manual LayerNorm over last dim (bd)
        mu = h.mean(dim=-1, keepdim=True)
        var = h.var(dim=-1, keepdim=True, correction=0)
        h = (h - mu) / (var + self.eps).sqrt()
        h = h * self.ln_w[None, :, None, :] + self.ln_b[None, :, None, :]

        # GELU + Dropout
        h = F.gelu(h)
        h = self.dropout(h)

        # ── 2) Attention logits: [B, P, S, H] ───────────────────────
        attn = (torch.einsum('bpsn,phn->bpsh', h, self.query_w)
                + self.query_b[None, :, None, :])
        attn = attn + self.pos_bias[:, :, :S].permute(0, 2, 1).unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, :, None], float('-inf'))

        attn = torch.softmax(attn, dim=2)                # [B, P, S, H]

        # ── 3) Pool first, then value-project (memory trick) ────────
        weighted_h = torch.einsum('bpsh,bpsn->bphn', attn, h)  # [B, P, H, bd]
        head_out = (torch.einsum('bphn,phkn->bphk', weighted_h, self.value_w)
                    + self.value_b[None, :, :, :])        # [B, P, H, hd]

        # ── 4) Concat heads → dropout → classify → scalar ───────────
        pooled = head_out.reshape(B, P, H * hd)
        pooled = self.dropout(pooled)

        out = (torch.einsum('bpv,pov->bpo', pooled, self.class_w)
               + self.class_b[None, :, :])                # [B, P, 1]
        return out.squeeze(-1)                             # [B, P]


# ── Batched training ─────────────────────────────────────────────────


def train_all_probes_batched(train_X_list, train_y,
                              test_X_list, test_y_np,
                              data_fractions, n_repeats, seed,
                              device, max_seq_len):
    """Train all data-scaling probes *simultaneously*.

    All P = len(data_fractions) * n_repeats BottleneckAttentionProbes share
    a single batched forward/backward pass.  A boolean membership mask routes
    different data subsets to different probes, with per-probe early stopping.
    Data stays on CPU and is padded per-batch to avoid OOM.
    """
    input_dim = train_X_list[0].shape[1]
    N_train = len(train_X_list)
    n_probes = len(data_fractions) * n_repeats

    train_y_t = torch.from_numpy(train_y).float().to(device)

    # ── Build membership masks: [P, N_train] ─────────────────────────
    membership = torch.zeros(n_probes, N_train, dtype=torch.bool, device=device)
    probe_idx = 0
    for frac in data_fractions:
        n_subset = max(5, int(N_train * frac))
        for rep in range(n_repeats):
            rng = np.random.default_rng(seed + rep)
            indices = rng.choice(N_train, n_subset, replace=False)
            membership[probe_idx, indices] = True
            probe_idx += 1

    # ── Create batched probes ─────────────────────────────────────────
    probes = BatchedBottleneckAttentionProbes(
        n_probes=n_probes,
        input_dim=input_dim,
        bottleneck_dim=BOTTLENECK_DIM,
        num_heads=NUM_HEADS,
        max_seq_len=max_seq_len,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(probes.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    # Per-probe early stopping state
    best_loss = torch.full((n_probes,), float("inf"))
    best_state = {n: p.data.clone() for n, p in probes.named_parameters()}
    no_improve = torch.zeros(n_probes, dtype=torch.long)
    active = torch.ones(n_probes, dtype=torch.bool)          # CPU
    active_dev = active.to(device)

    for epoch in range(EPOCHS):
        probes.train()
        perm = np.random.permutation(N_train)
        epoch_loss_sum = torch.zeros(n_probes, device=device)
        epoch_count = torch.zeros(n_probes, device=device)

        for start in range(0, N_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_train)
            b_idx = perm[start:end]

            x_batch, m_batch = _pad_batch(train_X_list, b_idx, device)
            y_batch = train_y_t[b_idx]

            # Forward: [B, P]
            preds = probes(x_batch, m_batch)

            # Per-element squared error: [B, P]
            sq_err = (preds - y_batch.unsqueeze(1)) ** 2

            # Membership + active mask
            b_idx_t = torch.from_numpy(b_idx).long().to(device)
            batch_member = membership[:, b_idx_t].T.float()     # [B, P]
            batch_mask = batch_member * active_dev.unsqueeze(0).float()

            counts = batch_mask.sum(dim=0).clamp(min=1)
            per_probe_loss = (sq_err * batch_mask).sum(dim=0) / counts
            loss = per_probe_loss.sum()

            optimizer.zero_grad()
            loss.backward()

            # Zero grads for frozen probes
            with torch.no_grad():
                inactive = ~active_dev
                if inactive.any():
                    for _, param in probes.named_parameters():
                        if param.grad is not None:
                            param.grad[inactive] = 0

            torch.nn.utils.clip_grad_norm_(probes.parameters(), GRAD_CLIP)
            optimizer.step()

            # Restore frozen probes (counteract weight-decay drift)
            with torch.no_grad():
                if inactive.any():
                    for name, param in probes.named_parameters():
                        param.data[inactive] = best_state[name][inactive].to(device)

            # Accumulate epoch loss
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
                for name, param in probes.named_parameters():
                    best_state[name][j] = param.data[j].clone()
                no_improve[j] = 0
            else:
                no_improve[j] += 1
            if no_improve[j] >= PATIENCE:
                active[j] = False
                for name, param in probes.named_parameters():
                    param.data[j] = best_state[name][j].to(device)

        active_dev = active.to(device)
        if not active.any():
            break

        if (epoch + 1) % 50 == 0:
            n_active = int(active.sum())
            print(f"  Epoch {epoch + 1}/{EPOCHS}, {n_active}/{n_probes} probes still active")

    # ── Restore best weights & evaluate ───────────────────────────────
    with torch.no_grad():
        for name, param in probes.named_parameters():
            param.data.copy_(best_state[name].to(device))

    probes.eval()
    train_y_np = train_y_t.cpu().numpy()
    N_test = len(test_X_list)

    # Batched eval to avoid OOM
    test_preds_parts = []
    with torch.no_grad():
        for start in range(0, N_test, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_test)
            x_b, m_b = _pad_batch(test_X_list, list(range(start, end)), device)
            test_preds_parts.append(probes(x_b, m_b).cpu().numpy())
    test_preds = np.concatenate(test_preds_parts, axis=0)  # [N_test, P]

    train_preds_parts = []
    with torch.no_grad():
        for start in range(0, N_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_train)
            x_b, m_b = _pad_batch(train_X_list, list(range(start, end)), device)
            train_preds_parts.append(probes(x_b, m_b).cpu().numpy())
    train_preds = np.concatenate(train_preds_parts, axis=0)  # [N_train, P]

    # Test R²
    ss_tot = np.sum((test_y_np - test_y_np.mean()) ** 2)
    r2_list = []
    for j in range(n_probes):
        ss_res = np.sum((test_y_np - test_preds[:, j]) ** 2)
        r2_list.append(float(1 - ss_res / max(ss_tot, 1e-8)))

    # Train R² (each probe evaluated on its own training subset)
    membership_np = membership.cpu().numpy()
    train_r2_list = []
    for j in range(n_probes):
        member_idx = membership_np[j]
        y_sub = train_y_np[member_idx]
        pred_sub = train_preds[member_idx, j]
        ss_tot_tr = np.sum((y_sub - y_sub.mean()) ** 2)
        ss_res_tr = np.sum((y_sub - pred_sub) ** 2)
        train_r2_list.append(float(1 - ss_res_tr / max(ss_tot_tr, 1e-8)))

    return r2_list, train_r2_list


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

    # ── Train all probes simultaneously ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(
        max(x.shape[0] for x in full_train_X),
        max(x.shape[0] for x in test_X),
    )
    n_probes = len(DATA_FRACTIONS) * N_REPEATS
    print(f"\nTraining {n_probes} probes simultaneously on {device} "
          f"({N_REPEATS} repeats × {len(DATA_FRACTIONS)} fractions, "
          f"max_seq_len={max_seq_len})...")
    r2_all, train_r2_all = train_all_probes_batched(
        full_train_X, full_train_y,
        test_X, test_y,
        DATA_FRACTIONS, N_REPEATS, SEED,
        device, max_seq_len,
    )

    # ── Reshape results ──────────────────────────────────────────────
    sizes = []
    mean_r2s = []
    std_r2s = []
    mean_train_r2s = []
    std_train_r2s = []

    for i, frac in enumerate(DATA_FRACTIONS):
        n_subset = max(5, int(n_train * frac))
        r2_scores = r2_all[i * N_REPEATS : (i + 1) * N_REPEATS]
        train_r2_scores = train_r2_all[i * N_REPEATS : (i + 1) * N_REPEATS]
        mean_r2 = float(np.mean(r2_scores))
        std_r2 = float(np.std(r2_scores))
        mean_tr = float(np.mean(train_r2_scores))
        std_tr = float(np.std(train_r2_scores))
        sizes.append(n_subset)
        mean_r2s.append(mean_r2)
        std_r2s.append(std_r2)
        mean_train_r2s.append(mean_tr)
        std_train_r2s.append(std_tr)
        print(f"  {frac*100:5.1f}% ({n_subset:4d} samples): Test R² = {mean_r2:.3f} ± {std_r2:.3f}  Train R² = {mean_tr:.3f} ± {std_tr:.3f}")

    # ── Save results ──────────────────────────────────────────────────
    output_dir = DATA_DIR / "data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "fractions": DATA_FRACTIONS,
        "sizes": sizes,
        "mean_r2": mean_r2s,
        "std_r2": std_r2s,
        "mean_train_r2": mean_train_r2s,
        "std_train_r2": std_train_r2s,
        "n_repeats": N_REPEATS,
        "n_train_full": n_train,
        "n_test": len(test_X),
    }
    with open(output_dir / "entropy_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    sizes_arr = np.array(sizes)
    test_mean = np.array(mean_r2s)
    test_std = np.array(std_r2s)
    train_mean = np.array(mean_train_r2s)
    train_std = np.array(std_train_r2s)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes_arr, train_mean, "r-o", markersize=5, label="Train R²")
    ax.fill_between(sizes_arr, train_mean - train_std, train_mean + train_std,
                    color="red", alpha=0.15)
    ax.plot(sizes_arr, test_mean, "b-o", markersize=5, label="Test R²")
    ax.fill_between(sizes_arr, test_mean - test_std, test_mean + test_std,
                    color="blue", alpha=0.15)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R²")
    ax.set_title("Entropy Probe: R² vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_data_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {output_dir / 'entropy_data_scaling.png'}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
