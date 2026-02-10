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


def prepad_to_gpu(X_list, device, target_max_len=None):
    """Pad all variable-length sequences and transfer to GPU in one shot."""
    hidden_dim = X_list[0].shape[1]
    max_len = target_max_len or max(x.shape[0] for x in X_list)
    N = len(X_list)

    X_pad = torch.zeros(N, max_len, hidden_dim, device=device)
    mask = torch.zeros(N, max_len, dtype=torch.bool, device=device)
    for i, x in enumerate(X_list):
        sl = x.shape[0]
        X_pad[i, :sl] = torch.from_numpy(x).to(device)
        mask[i, :sl] = True

    return X_pad, mask


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


def train_all_probes_batched(train_X_pad, train_mask, train_y_t,
                              test_X_pad, test_mask, test_y_np,
                              data_fractions, n_repeats, seed,
                              device, max_seq_len):
    """Train all data-scaling probes *simultaneously*.

    All P = len(data_fractions) * n_repeats probes share a single batched
    forward pass.  A boolean membership mask routes different data subsets
    to different probes.
    """
    hidden_dim = train_X_pad.shape[2]
    N_train = train_X_pad.shape[0]
    n_probes = len(data_fractions) * n_repeats

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
        perm = torch.randperm(N_train, device=device)

        for start in range(0, N_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_train)
            b_idx = perm[start:end]

            x_batch = train_X_pad[b_idx]        # [B, S, D]
            m_batch = train_mask[b_idx]          # [B, S]
            y_batch = train_y_t[b_idx]           # [B]

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
            batch_member = membership[:, b_idx].T.float()   # [B, P]
            counts = batch_member.sum(dim=0).clamp(min=1)
            per_probe_loss = (weighted_nll * batch_member).sum(dim=0) / counts
            loss = per_probe_loss.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), GRAD_CLIP)
            optimizer.step()

    # ── Evaluate ──────────────────────────────────────────────────────
    probes.eval()
    with torch.no_grad():
        test_logits = probes(test_X_pad, test_mask)         # [N_test, P, C]
        test_preds = test_logits.argmax(dim=-1).cpu().numpy()  # [N_test, P]

    f1_list = []
    for p in range(n_probes):
        f1_list.append(_compute_f1(test_y_np, test_preds[:, p]))

    return f1_list


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

    # ── Pre-pad and transfer all data to GPU once ─────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(
        max(x.shape[0] for x in full_train_X),
        max(x.shape[0] for x in test_X),
    )
    print(f"\nPre-padding and transferring data to {device} (max_seq_len={max_seq_len})...")
    train_X_pad, train_mask = prepad_to_gpu(full_train_X, device)
    test_X_pad, test_mask = prepad_to_gpu(test_X, device)
    train_y_t = torch.from_numpy(full_train_y).long().to(device)
    test_y_np = test_y  # keep numpy for F1 computation
    print(f"  Train: {train_X_pad.shape}, Test: {test_X_pad.shape}")

    # ── Train all probes simultaneously ───────────────────────────────
    n_probes = len(DATA_FRACTIONS) * N_REPEATS
    print(f"\nTraining {n_probes} probes simultaneously ({N_REPEATS} repeats × {len(DATA_FRACTIONS)} fractions)...")
    f1_all = train_all_probes_batched(
        train_X_pad, train_mask, train_y_t,
        test_X_pad, test_mask, test_y_np,
        DATA_FRACTIONS, N_REPEATS, SEED,
        device, max_seq_len,
    )

    # ── Reshape results ───────────────────────────────────────────────
    sizes = []
    mean_f1s = []
    std_f1s = []

    for i, frac in enumerate(DATA_FRACTIONS):
        n_subset = max(2, int(n_train * frac))
        f1_scores = f1_all[i * N_REPEATS : (i + 1) * N_REPEATS]
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
