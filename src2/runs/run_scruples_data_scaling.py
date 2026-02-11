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
import torch.nn.functional as F

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
LR = 8e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
EPOCHS = 40
BATCH_SIZE = 64
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


# ── Batched attention probes ─────────────────────────────────────────


class BatchedAttentionProbes(nn.Module):
    """P attention-pooling probes trained simultaneously via batched einsum.

    Mathematically equivalent to P independent ``AttentionPoolingProbe``s, but
    all weights are stacked along a leading *probe* dimension so every training
    step is a single set of fused matmuls.

    Memory trick: because the value projection is linear, we can **pool first,
    then project**.  Instead of materialising ``[B, P, S, V]`` (huge), we
    compute the attention-weighted mean of the raw input ``[B, P, H, D]``
    (small) and then apply the per-head value projection.  This saves ~100×
    memory while giving identical results.
    """

    def __init__(self, n_probes, hidden_dim, num_heads, output_dim,
                 max_seq_len, dropout):
        super().__init__()
        self.n_probes = n_probes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.output_dim = output_dim

        # query_proj  – Linear(D, H) per probe
        self.query_w = nn.Parameter(torch.empty(n_probes, num_heads, hidden_dim))
        self.query_b = nn.Parameter(torch.zeros(n_probes, num_heads))

        # position bias per probe
        self.pos_bias = nn.Parameter(torch.zeros(n_probes, num_heads, max_seq_len))

        # value_proj  – stored as [P, H, head_dim, D] for the pool-first trick
        self.value_w = nn.Parameter(
            torch.empty(n_probes, num_heads, self.head_dim, hidden_dim))
        self.value_b = nn.Parameter(
            torch.zeros(n_probes, num_heads, self.head_dim))

        # classifier  – Linear(H*head_dim, output_dim) per probe
        self.class_w = nn.Parameter(
            torch.empty(n_probes, output_dim, num_heads * self.head_dim))
        self.class_b = nn.Parameter(torch.zeros(n_probes, output_dim))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        """Match ``nn.Linear`` default init for each probe independently."""
        for p in range(self.n_probes):
            # query_proj: weight [H, D], bias [H]
            nn.init.kaiming_uniform_(self.query_w.data[p], a=5 ** 0.5)
            bound = 1.0 / self.hidden_dim ** 0.5
            nn.init.uniform_(self.query_b.data[p], -bound, bound)

            # value_proj: logical weight is [H*hd, D]
            w_flat = self.value_w.data[p].view(-1, self.hidden_dim)
            nn.init.kaiming_uniform_(w_flat, a=5 ** 0.5)
            # (view mutation writes through to self.value_w.data[p])
            nn.init.uniform_(
                self.value_b.data[p].view(-1),
                -bound, bound,
            )

            # classifier: weight [O, V], bias [O]
            nn.init.kaiming_uniform_(self.class_w.data[p], a=5 ** 0.5)
            fan_in = self.num_heads * self.head_dim
            bound_c = 1.0 / fan_in ** 0.5
            nn.init.uniform_(self.class_b.data[p], -bound_c, bound_c)

    # --------------------------------------------------------------- forward
    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, S, D]
            mask: [B, S] boolean (True = valid)
        Returns:
            [B, P, output_dim]
        """
        B, S, D = x.shape
        P = self.n_probes
        H = self.num_heads
        hd = self.head_dim

        # 1) Attention logits  [B, P, S, H]
        attn = (torch.einsum('bsd,phd->bpsh', x, self.query_w)
                + self.query_b[None, :, None, :])

        # position bias  [P, H, S] -> [1, P, S, H]
        attn = attn + self.pos_bias[:, :, :S].permute(0, 2, 1).unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, :, None], float('-inf'))

        attn = torch.softmax(attn, dim=2)               # [B, P, S, H]

        # 2) Attention-weighted mean of x  (pool BEFORE value proj – saves mem)
        #    [B,P,S,H] × [B,S,D] -> [B, P, H, D]
        weighted_x = torch.einsum('bpsh,bsd->bphd', attn, x)

        # 3) Per-head value projection  [B, P, H, D] × [P, H, hd, D] -> [B, P, H, hd]
        head_out = (torch.einsum('bphd,phkd->bphk', weighted_x, self.value_w)
                    + self.value_b[None, :, :, :])

        # 4) Concat heads -> dropout -> classify
        pooled = head_out.reshape(B, P, H * hd)          # [B, P, V]
        pooled = self.dropout(pooled)

        logits = (torch.einsum('bpv,pov->bpo', pooled, self.class_w)
                  + self.class_b[None, :, :])
        return logits                                     # [B, P, output_dim]


# ── Batched training ─────────────────────────────────────────────────


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


def train_all_probes_batched(train_X_list, train_y,
                              test_X_list, test_y_np,
                              data_fractions, n_repeats, seed,
                              device, max_seq_len):
    """Train all data-scaling probes *simultaneously*.

    All P = len(data_fractions) * n_repeats probes share a single batched
    forward pass.  A boolean membership mask routes different data subsets
    to different probes.  Data stays on CPU and is padded per-batch to
    avoid OOM.
    """
    hidden_dim = train_X_list[0].shape[1]
    N_train = len(train_X_list)
    n_probes = len(data_fractions) * n_repeats

    train_y_t = torch.from_numpy(train_y).long().to(device)

    # ── Build membership masks: [P, N_train] ─────────────────────────
    membership = torch.zeros(n_probes, N_train, dtype=torch.bool, device=device)
    idx = 0
    for frac in data_fractions:
        n_sub = max(2, int(N_train * frac))
        for rep in range(n_repeats):
            rng = np.random.default_rng(seed + rep)
            indices = rng.choice(N_train, n_sub, replace=False)
            membership[idx, indices] = True
            idx += 1

    # Per-probe class weights: [P, C]
    class_weights_all = torch.zeros(n_probes, NUM_CLASSES, device=device)
    for p in range(n_probes):
        sub_y = train_y_t[membership[p]]
        counts = torch.bincount(sub_y, minlength=NUM_CLASSES).float().clamp(min=1)
        n = membership[p].sum().float()
        class_weights_all[p] = n / (NUM_CLASSES * counts)

    # ── Create batched probes ─────────────────────────────────────────
    probes = BatchedAttentionProbes(
        n_probes=n_probes,
        hidden_dim=hidden_dim,
        num_heads=NUM_HEADS,
        output_dim=NUM_CLASSES,
        max_seq_len=max_seq_len,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(probes.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        probes.train()
        perm = np.random.permutation(N_train)

        for start in range(0, N_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_train)
            b_idx = perm[start:end]

            x_batch, m_batch = _pad_batch(train_X_list, b_idx, device)
            y_batch = train_y_t[b_idx]

            # Forward: [B, P, C]
            logits = probes(x_batch, m_batch)

            # Per-sample per-probe NLL: [B, P]
            log_probs = F.log_softmax(logits, dim=-1)
            targets_exp = y_batch[:, None, None].expand(-1, n_probes, 1)
            nll = -log_probs.gather(dim=-1, index=targets_exp).squeeze(-1)

            # Apply per-probe class weighting
            sample_w = class_weights_all[:, y_batch].T     # [B, P]
            weighted_nll = nll * sample_w

            # Membership mask
            b_idx_t = torch.from_numpy(b_idx).long().to(device)
            batch_member = membership[:, b_idx_t].T.float()   # [B, P]
            counts = batch_member.sum(dim=0).clamp(min=1)
            per_probe_loss = (weighted_nll * batch_member).sum(dim=0) / counts
            loss = per_probe_loss.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), GRAD_CLIP)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}")

    # ── Evaluate (batched to avoid OOM) ───────────────────────────────
    probes.eval()
    N_test = len(test_X_list)
    test_pred_parts = []
    with torch.no_grad():
        for start in range(0, N_test, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_test)
            b_idx = list(range(start, end))
            x_batch, m_batch = _pad_batch(test_X_list, b_idx, device)
            logits = probes(x_batch, m_batch)  # [B, P, C]
            test_pred_parts.append(logits.argmax(dim=-1).cpu().numpy())
    test_preds = np.concatenate(test_pred_parts, axis=0)  # [N_test, P]

    train_pred_parts = []
    with torch.no_grad():
        for start in range(0, N_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_train)
            b_idx = list(range(start, end))
            x_batch, m_batch = _pad_batch(train_X_list, b_idx, device)
            logits = probes(x_batch, m_batch)
            train_pred_parts.append(logits.argmax(dim=-1).cpu().numpy())
    train_preds = np.concatenate(train_pred_parts, axis=0)  # [N_train, P]

    # Test metrics
    test_f1_list = []
    test_acc_list = []
    for p in range(n_probes):
        test_f1_list.append(_compute_f1(test_y_np, test_preds[:, p]))
        test_acc_list.append(float((test_y_np == test_preds[:, p]).mean()))

    # Train metrics (each probe on its own subset)
    membership_np = membership.cpu().numpy()
    train_y_np = train_y
    train_f1_list = []
    train_acc_list = []
    for p in range(n_probes):
        member_idx = membership_np[p]
        y_sub = train_y_np[member_idx]
        pred_sub = train_preds[member_idx, p]
        train_f1_list.append(_compute_f1(y_sub, pred_sub))
        train_acc_list.append(float((y_sub == pred_sub).mean()))

    return test_f1_list, train_f1_list, test_acc_list, train_acc_list


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
        data_slice=split_info,
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

    # ── Fixed train/test split (canonical from get_uncertainty_robust_split) ──
    train_aids = set(split_info.train_df["anecdote_id"].unique()) | set(split_info.val_df["anecdote_id"].unique())  # merge val into train for scaling
    test_aids = set(split_info.test_df["anecdote_id"].unique())

    train_idx = [i for i, a in enumerate(anecdote_ids) if a in train_aids]
    test_idx = [i for i, a in enumerate(anecdote_ids) if a in test_aids]

    full_train_X = [X_list[i] for i in train_idx]
    full_train_y = y[train_idx]
    test_X = [X_list[i] for i in test_idx]
    test_y = y[test_idx]
    n_train = len(full_train_X)
    print(f"Train: {n_train}, Test: {len(test_X)}")

    # ── Train all probes simultaneously ───────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(
        max(x.shape[0] for x in full_train_X),
        max(x.shape[0] for x in test_X),
    )
    n_probes = len(DATA_FRACTIONS) * N_REPEATS
    print(f"\nTraining {n_probes} probes simultaneously on {device} "
          f"({N_REPEATS} repeats × {len(DATA_FRACTIONS)} fractions, "
          f"max_seq_len={max_seq_len})...")
    test_f1_all, train_f1_all, test_acc_all, train_acc_all = train_all_probes_batched(
        full_train_X, full_train_y,
        test_X, test_y,
        DATA_FRACTIONS, N_REPEATS, SEED,
        device, max_seq_len,
    )

    # ── Reshape results ───────────────────────────────────────────────
    sizes = []
    mean_f1s, std_f1s = [], []
    mean_train_f1s, std_train_f1s = [], []
    mean_test_accs, std_test_accs = [], []
    mean_train_accs, std_train_accs = [], []

    for i, frac in enumerate(DATA_FRACTIONS):
        n_subset = max(2, int(n_train * frac))
        sl = slice(i * N_REPEATS, (i + 1) * N_REPEATS)

        sizes.append(n_subset)
        mean_f1s.append(float(np.mean(test_f1_all[sl])))
        std_f1s.append(float(np.std(test_f1_all[sl])))
        mean_train_f1s.append(float(np.mean(train_f1_all[sl])))
        std_train_f1s.append(float(np.std(train_f1_all[sl])))
        mean_test_accs.append(float(np.mean(test_acc_all[sl])))
        std_test_accs.append(float(np.std(test_acc_all[sl])))
        mean_train_accs.append(float(np.mean(train_acc_all[sl])))
        std_train_accs.append(float(np.std(train_acc_all[sl])))

        print(f"  {frac*100:5.1f}% ({n_subset:4d} samples): "
              f"Test F1={mean_f1s[-1]:.3f}±{std_f1s[-1]:.3f}  "
              f"Train F1={mean_train_f1s[-1]:.3f}±{std_train_f1s[-1]:.3f}  "
              f"Test Acc={mean_test_accs[-1]:.3f}  Train Acc={mean_train_accs[-1]:.3f}")

    # ── Save results ──────────────────────────────────────────────────
    output_dir = DATA_DIR / "data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "fractions": DATA_FRACTIONS,
        "sizes": sizes,
        "mean_f1": mean_f1s,
        "std_f1": std_f1s,
        "mean_train_f1": mean_train_f1s,
        "std_train_f1": std_train_f1s,
        "mean_test_acc": mean_test_accs,
        "std_test_acc": std_test_accs,
        "mean_train_acc": mean_train_accs,
        "std_train_acc": std_train_accs,
        "n_repeats": N_REPEATS,
        "n_train_full": n_train,
        "n_test": len(test_X),
    }
    with open(output_dir / "scruples_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    sizes_arr = np.array(sizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 subplot
    ax = axes[0]
    tr_f1 = np.array(mean_train_f1s)
    tr_f1_s = np.array(std_train_f1s)
    te_f1 = np.array(mean_f1s)
    te_f1_s = np.array(std_f1s)
    ax.plot(sizes_arr, tr_f1, "r-o", markersize=5, label="Train F1")
    ax.fill_between(sizes_arr, tr_f1 - tr_f1_s, tr_f1 + tr_f1_s, color="red", alpha=0.15)
    ax.plot(sizes_arr, te_f1, "b-o", markersize=5, label="Test F1")
    ax.fill_between(sizes_arr, te_f1 - te_f1_s, te_f1 + te_f1_s, color="blue", alpha=0.15)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1")
    ax.set_title("Sycophancy Probe: F1 vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy subplot
    ax = axes[1]
    tr_acc = np.array(mean_train_accs)
    tr_acc_s = np.array(std_train_accs)
    te_acc = np.array(mean_test_accs)
    te_acc_s = np.array(std_test_accs)
    ax.plot(sizes_arr, tr_acc, "r-o", markersize=5, label="Train Accuracy")
    ax.fill_between(sizes_arr, tr_acc - tr_acc_s, tr_acc + tr_acc_s, color="red", alpha=0.15)
    ax.plot(sizes_arr, te_acc, "b-o", markersize=5, label="Test Accuracy")
    ax.fill_between(sizes_arr, te_acc - te_acc_s, te_acc + te_acc_s, color="blue", alpha=0.15)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sycophancy Probe: Accuracy vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "scruples_data_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {output_dir / 'scruples_data_scaling.png'}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
