#!/usr/bin/env python3
"""
Entropy attention probe.

Trains an attention probe that predicts the entropy of the answer probability
distribution (A/B/C/D) from model activations at forcing boundaries.

Instead of predicting the full 4-class distribution (which overfits), we predict
a single scalar: H(p) = -sum(p_i * log(p_i)), the Shannon entropy of the
answer distribution. This is a simpler regression target that captures how
"uncertain" the model is at each forcing point.

Uses the same data pipeline as run_answer_probe.py (forced response activations,
sentence sampling, CoT trimming, mean subtraction).

Usage:
    python -m src2.runs.run_entropy_probe
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src2.data_slice import DataSlice
from src2.methods.attention_probe import AttentionPoolingProbe
from src2.tasks.forced_response.task import ForcingTask
from src2.utils.questions import load_gpqa_from_huggingface

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

# Custom questions always go to train
CUSTOM_TRAIN_IDS = ["custom_bagel_001", "starfish", "waffle"]

# GPQA Diamond: load N total, split into train/eval
NUM_GPQA_TRAIN = 50
NUM_GPQA_EVAL = 10

# Extra eval-only questions
EXTRA_EVAL_IDS = ["blackmail_mc_001", "blackmail_ab_001"]

# Sentence sampling
MAX_SENTENCES_PER_QUESTION_TRAIN = 50
MAX_SENTENCES_PER_QUESTION_EVAL = 50
ANSWER_LABELS = ["A", "B", "C", "D"]

# Training hyperparameters
BOTTLENECK_DIM = 8
FREEZE_PROJECTION = False
USE_PCA = False
MEAN_SUBTRACT = True  # remove question identity → force within-question generalization
USE_LINEAR_BASELINE = True  # mean-pool + linear instead of attention probe
NUM_HEADS = 1
LR = 1e-3
EPOCHS = 500
BATCH_SIZE = 256
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.05
DROPOUT = 0.5
SEED = 42

# Validation / early stopping
VAL_SPLIT = 0.2
PATIENCE = 50
MIN_DELTA = 0.001  # smaller delta for regression (MSE scale is smaller)

EXTRACT_ACTIVATIONS = False
TOKEN_POSITION = "full_sequence"
TRIM_TO_COT = True


# ── Helpers ───────────────────────────────────────────────────────────


def sample_sentence_indices(
    forcing_dir: Path,
    question_ids: list,
    max_per_question: int,
    seed: int,
) -> dict:
    """For each question, find available sentence indices and randomly sample N."""
    rng = np.random.default_rng(seed)
    sentence_map = {}
    for qid in question_ids:
        summaries = sorted(
            forcing_dir.glob(f"{qid}/rollout_*/*/sentence_*/summary.json")
        )
        available = {int(p.parent.name.split("_")[1]) for p in summaries}
        if len(available) <= max_per_question:
            sentence_map[qid] = available
        else:
            sentence_map[qid] = set(
                rng.choice(sorted(available), max_per_question, replace=False)
            )
    return sentence_map


def dist_dict_to_entropy(dist_dict: dict) -> float:
    """Convert {"A": 0.4, "B": 0.28, ...} → scalar entropy H(p).

    Only includes A/B/C/D keys. Re-normalizes to sum to 1.
    Returns None if no ABCD keys are present.
    """
    vals = [dist_dict.get(label, 0.0) for label in ANSWER_LABELS]
    total = sum(vals)
    if total == 0:
        return None
    probs = np.array([v / total for v in vals], dtype=np.float64)
    # Shannon entropy: H = -sum(p * log(p)), with 0*log(0) = 0
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log(p)
    return float(entropy)


def _compute_base_prompt_tokens(full_prompt: str, tokenizer) -> int:
    """Find number of tokens in the base prompt (everything up to and including <think>)."""
    think_tag = "<think>"
    pos = full_prompt.find(think_tag)
    if pos < 0:
        return 0
    end = pos + len(think_tag)
    if end < len(full_prompt) and full_prompt[end] == "\n":
        end += 1
    base_text = full_prompt[:end]
    ids = tokenizer(base_text, truncation=True, max_length=4096)["input_ids"]
    return len(ids)


def _load_question_data(
    forcing: ForcingTask,
    qid: str,
    sentence_indices,
    layer: int,
    token_position: str,
    tokenizer,
    trim_to_cot: bool,
) -> list:
    """Load data for a single question. Designed to run in a thread."""
    ds = DataSlice(ids={qid}, sentence_indices=sentence_indices)
    samples = forcing.get_probe_data(layer, ds, token_position)

    base_prompt_tokens = None
    results = []
    for sample in samples:
        entropy = dist_dict_to_entropy(sample["answer_distribution"])
        if entropy is None:
            continue

        act = sample["activation"]

        if trim_to_cot and tokenizer is not None and act.ndim == 2:
            if base_prompt_tokens is None:
                base_prompt_tokens = _compute_base_prompt_tokens(
                    sample["full_prompt"], tokenizer,
                )
            if base_prompt_tokens > 0 and act.shape[0] > base_prompt_tokens:
                act = act[base_prompt_tokens:]

        results.append((act, entropy, sample["question_id"], sample["sentence_idx"]))
    return results


def load_probe_data(
    forcing: ForcingTask,
    question_ids: list,
    sentence_map: dict,
    layer: int,
    token_position: str,
    tokenizer=None,
    trim_to_cot: bool = False,
    max_workers: int = 8,
) -> dict:
    """Load activations + entropy targets for the given questions.

    Parallelizes across questions using threads (I/O bound).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    X_list = []
    y_entropy_list = []
    q_ids = []
    s_indices = []

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for qid in question_ids:
            fut = pool.submit(
                _load_question_data,
                forcing, qid, sentence_map.get(qid),
                layer, token_position, tokenizer, trim_to_cot,
            )
            futures[fut] = qid

        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 10 == 0:
                print(f"    Loaded {done}/{len(question_ids)} questions...")
            for act, entropy, question_id, sentence_idx in fut.result():
                X_list.append(act)
                y_entropy_list.append(entropy)
                q_ids.append(question_id)
                s_indices.append(sentence_idx)

    y_entropy = np.array(y_entropy_list, dtype=np.float32) if y_entropy_list else np.zeros(0, dtype=np.float32)

    return {
        "X_list": X_list,
        "y_entropy": y_entropy,
        "question_ids": q_ids,
        "sentence_indices": s_indices,
    }


def mean_subtract_per_question(data: dict) -> dict:
    """Subtract per-question mean activation from every token in every sample."""
    qid_to_indices = defaultdict(list)
    for i, qid in enumerate(data["question_ids"]):
        qid_to_indices[qid].append(i)

    question_means = {}
    for qid, indices in qid_to_indices.items():
        all_tokens = np.concatenate([data["X_list"][i] for i in indices], axis=0)
        question_means[qid] = all_tokens.mean(axis=0)

    for i, qid in enumerate(data["question_ids"]):
        data["X_list"][i] = data["X_list"][i] - question_means[qid]

    data["_question_means"] = question_means
    return data


def fit_pca(train_data: dict, n_components: int) -> dict:
    """Fit PCA on train token activations, return projection info.

    Concatenates all tokens from all train samples, fits PCA, returns
    the mean and components needed to transform any data.
    """
    from sklearn.decomposition import IncrementalPCA

    print(f"  Fitting PCA ({n_components} components) on train activations...")
    # Use IncrementalPCA to avoid memory issues with large datasets
    pca = IncrementalPCA(n_components=n_components, batch_size=4096)
    # Fit on all train tokens
    all_tokens = np.concatenate(train_data["X_list"], axis=0)
    print(f"    Total tokens for PCA fit: {all_tokens.shape[0]:,} × {all_tokens.shape[1]}")
    pca.fit(all_tokens)
    explained = pca.explained_variance_ratio_.sum()
    print(f"    Explained variance: {explained:.4f} ({n_components} components)")
    return pca


def apply_pca(data: dict, pca) -> dict:
    """Transform activations using fitted PCA. Modifies data in-place."""
    data["X_list"] = [pca.transform(x).astype(np.float32) for x in data["X_list"]]
    return data


# ── Probe wrappers ────────────────────────────────────────────────


class FrozenProjectionProbe(nn.Module):
    """Fixed random projection → attention pooling → regression head."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float,
        seed: int = 42,
    ):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(input_dim, proj_dim, generator=gen) / (proj_dim ** 0.5)
        self.register_buffer("proj_weight", W)

        self.norm = nn.LayerNorm(proj_dim)
        self.attn_probe = AttentionPoolingProbe(
            hidden_dim=proj_dim,
            num_heads=num_heads,
            output_dim=1,  # scalar entropy
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        x = x @ self.proj_weight
        x = self.norm(x)
        return self.attn_probe(x, mask).squeeze(-1)  # [batch]


class MeanPoolLinearProbe(nn.Module):
    """Mean-pool over sequence → dropout → linear → scalar. Simplest baseline."""

    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, hidden_dim], mask: [batch, seq_len]
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = x.sum(dim=1) / lengths  # [batch, hidden_dim]
        else:
            pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.linear(pooled).squeeze(-1)  # [batch]


class BottleneckAttentionProbe(nn.Module):
    """Learned projection → attention pooling → regression head."""

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_probe = AttentionPoolingProbe(
            hidden_dim=bottleneck_dim,
            num_heads=num_heads,
            output_dim=1,  # scalar entropy
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        x = self.proj(x)
        return self.attn_probe(x, mask).squeeze(-1)  # [batch]


# ── Training ──────────────────────────────────────────────────────────


def _eval_loss_and_metrics(probe, X_list, y_entropy, device, batch_size=64):
    """Compute MSE loss, R², and Pearson correlation without gradients."""
    probe.eval()
    hidden_dim = X_list[0].shape[1]
    all_preds = []
    all_targets = []

    for start in range(0, len(X_list), batch_size):
        end = min(start + batch_size, len(X_list))
        batch_X = X_list[start:end]
        batch_y = y_entropy[start:end]
        batch_max_len = max(x.shape[0] for x in batch_X)

        X_pad = torch.zeros(len(batch_X), batch_max_len, hidden_dim, device=device)
        mask = torch.zeros(len(batch_X), batch_max_len, dtype=torch.bool, device=device)
        for i, x in enumerate(batch_X):
            sl = x.shape[0]
            X_pad[i, :sl, :] = torch.from_numpy(x).to(device)
            mask[i, :sl] = True
        y_t = torch.from_numpy(batch_y).float().to(device)

        with torch.no_grad():
            pred = probe(X_pad, mask)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_y)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = float(np.mean((preds - targets) ** 2))
    # R²
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    return mse, float(r2)


def _predict(probe, X_list, device):
    """Get predicted entropy values for a list of activations."""
    probe.eval()
    preds = []
    for x in X_list:
        seq_len = x.shape[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        with torch.no_grad():
            pred = probe(x_t, m)
        preds.append(pred.item())
    return np.array(preds, dtype=np.float32)


def train_and_evaluate(
    train_X: list,
    train_y: np.ndarray,
    val_X: list,
    val_y: np.ndarray,
    test_X: list,
    test_y: np.ndarray,
    num_heads: int = NUM_HEADS,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train entropy probe with MSE loss, early stopping on validation set."""
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}, batch_size: {batch_size}")

    hidden_dim = train_X[0].shape[1]
    all_X = list(train_X) + list(val_X) + list(test_X)
    max_seq_len = max(x.shape[0] for x in all_X) if all_X else 1
    n_samples = len(train_X)

    if USE_LINEAR_BASELINE:
        probe = MeanPoolLinearProbe(
            input_dim=hidden_dim,
            dropout=DROPOUT,
        ).to(device)
        proj_label = "mean-pool linear"
    elif USE_PCA:
        # Data already PCA'd — just attention probe, no projection layer
        probe = AttentionPoolingProbe(
            hidden_dim=hidden_dim,  # already BOTTLENECK_DIM after PCA
            num_heads=num_heads,
            output_dim=1,
            max_seq_len=max_seq_len,
            dropout=DROPOUT,
        ).to(device)
        # Wrap forward to squeeze output
        _orig_forward = probe.forward
        def _squeezed_forward(x, mask=None):
            return _orig_forward(x, mask).squeeze(-1)
        probe.forward = _squeezed_forward
        proj_label = "PCA"
    elif FREEZE_PROJECTION:
        probe = FrozenProjectionProbe(
            input_dim=hidden_dim,
            proj_dim=BOTTLENECK_DIM,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=DROPOUT,
            seed=SEED,
        ).to(device)
        proj_label = "frozen"
    else:
        probe = BottleneckAttentionProbe(
            input_dim=hidden_dim,
            bottleneck_dim=BOTTLENECK_DIM,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=DROPOUT,
        ).to(device)
        proj_label = "learned"

    n_trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in probe.parameters())
    print(f"  Probe: {n_trainable:,} trainable / {n_total:,} total params ({proj_label} projection → {hidden_dim}d input)")

    # Print target stats
    print(f"  Entropy target stats: mean={train_y.mean():.4f}, std={train_y.std():.4f}, "
          f"min={train_y.min():.4f}, max={train_y.max():.4f}")

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )
    loss_fn = nn.MSELoss()

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    # Track metrics for plotting
    history = {"epoch": [], "train_r2": [], "val_r2": []}

    for epoch in range(epochs):
        # ── Train ──
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
            mask = torch.zeros(
                len(batch_X), batch_max_len, dtype=torch.bool, device=device
            )
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

        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        if val_X:
            val_mse, val_r2 = _eval_loss_and_metrics(probe, val_X, val_y, device)
        else:
            val_mse, val_r2 = train_loss, 0.0

        scheduler.step()

        # ── Early stopping ──
        if val_mse < best_val_loss - MIN_DELTA:
            best_val_loss = val_mse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(probe.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            train_mse, train_r2 = _eval_loss_and_metrics(probe, train_X, train_y, device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}/{epochs}  "
                f"train: MSE={train_mse:.6f} R²={train_r2:.3f}  "
                f"val: MSE={val_mse:.6f} R²={val_r2:.3f}  "
                f"best_val: {best_val_loss:.6f} (ep {best_epoch})  "
                f"lr: {current_lr:.2e}"
            )

            if (epoch + 1) % 10 == 0:
                history["epoch"].append(epoch + 1)
                history["train_r2"].append(train_r2)
                history["val_r2"].append(val_r2)

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1} (no val improvement for {PATIENCE} epochs)")
            break

    # Restore best model
    if best_state is not None:
        probe.load_state_dict(best_state)
        print(f"  Restored best model from epoch {best_epoch} (val_MSE={best_val_loss:.6f})")

    # ── Predictions ──
    return {
        "test_preds": _predict(probe, test_X, device) if test_X else np.zeros(0),
        "train_preds": _predict(probe, train_X, device),
        "val_preds": _predict(probe, val_X, device) if val_X else np.zeros(0),
        "final_train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "history": history,
    }


# ── Metrics ───────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    question_ids: list,
) -> dict:
    """Compute MSE, R², Pearson correlation, and per-question breakdown."""
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0}

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))

    # Pearson correlation
    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    # Baseline: always predict mean
    mean_pred = np.full_like(y_true, y_true.mean())
    baseline_mse = float(np.mean((y_true - mean_pred) ** 2))

    # Per-question breakdown
    per_question = {}
    unique_qids = sorted(set(question_ids))
    for qid in unique_qids:
        idx = [i for i, q in enumerate(question_ids) if q == qid]
        q_true = y_true[idx]
        q_pred = y_pred[idx]
        q_mse = float(np.mean((q_true - q_pred) ** 2))
        q_mae = float(np.mean(np.abs(q_true - q_pred)))
        # Per-question R²
        q_ss_res = np.sum((q_true - q_pred) ** 2)
        q_ss_tot = np.sum((q_true - q_true.mean()) ** 2)
        q_r2 = float(1 - q_ss_res / max(q_ss_tot, 1e-8)) if q_ss_tot > 1e-8 else 0.0
        per_question[qid] = {
            "n_samples": len(idx),
            "mse": q_mse,
            "mae": q_mae,
            "r2": q_r2,
            "true_mean": float(q_true.mean()),
            "true_std": float(q_true.std()),
            "pred_mean": float(q_pred.mean()),
        }

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "baseline_mse": baseline_mse,
        "n_samples": n,
        "target_mean": float(y_true.mean()),
        "target_std": float(y_true.std()),
        "pred_mean": float(y_pred.mean()),
        "pred_std": float(y_pred.std()),
        "per_question": per_question,
    }


def print_results(metrics: dict, label: str = "Results"):
    """Pretty-print metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    if metrics.get("n_samples", 0) == 0:
        print("  No samples.")
        return

    print(f"  MSE:             {metrics['mse']:.6f}")
    print(f"  MAE:             {metrics['mae']:.4f}")
    print(f"  R²:              {metrics['r2']:.4f}")
    print(f"  Pearson r:       {metrics['pearson_r']:.4f}")
    print(f"  Baseline MSE:    {metrics['baseline_mse']:.6f}  (always predict mean)")
    print(f"  Target:          mean={metrics['target_mean']:.4f}, std={metrics['target_std']:.4f}")
    print(f"  Predicted:       mean={metrics['pred_mean']:.4f}, std={metrics['pred_std']:.4f}")
    print(f"  N samples:       {metrics['n_samples']}")

    if "per_question" in metrics:
        print(f"\n  Per-question breakdown:")
        for qid, vals in metrics["per_question"].items():
            print(
                f"    {qid:>25s}: "
                f"MSE={vals['mse']:.6f}  "
                f"MAE={vals['mae']:.4f}  "
                f"R²={vals['r2']:.3f}  "
                f"true={vals['true_mean']:.3f}±{vals['true_std']:.3f}  "
                f"pred={vals['pred_mean']:.3f}  "
                f"n={vals['n_samples']}"
            )


# ── Main ──────────────────────────────────────────────────────────────


def _compute_question_mean_entropy(forcing_dir: Path, question_ids: list) -> dict:
    """Compute mean entropy per question from forcing summaries on disk."""
    question_mean_entropy = {}
    for qid in question_ids:
        summaries = sorted(
            forcing_dir.glob(f"{qid}/rollout_*/*/sentence_*/summary.json")
        )
        entropies = []
        for sp in summaries:
            with open(sp) as f:
                s = json.load(f)
            ad = s.get("answer_distribution", {})
            ent = dist_dict_to_entropy(ad)
            if ent is not None:
                entropies.append(ent)
        question_mean_entropy[qid] = np.mean(entropies) if entropies else 0.0
    return question_mean_entropy


def _stratified_interleave(ids: list, entropy_map: dict, n_splits: int) -> list:
    """Sort IDs by entropy, then deal them round-robin into n_splits buckets.

    Returns list of n_splits lists, each covering the full entropy range.
    """
    sorted_ids = sorted(ids, key=lambda qid: entropy_map.get(qid, 0.0))
    buckets = [[] for _ in range(n_splits)]
    for i, qid in enumerate(sorted_ids):
        buckets[i % n_splits].append(qid)
    return buckets


def build_question_splits(forcing_dir: Path, seed: int = SEED) -> dict:
    """Build train/val/eval splits, stratified by mean entropy.

    All three splits cover the full entropy range via interleaved assignment.
    Custom and extra eval IDs are assigned first, then GPQA questions are
    stratified across remaining slots.
    """
    rng = np.random.default_rng(seed)

    gpqa_questions = load_gpqa_from_huggingface(
        subset="gpqa_diamond", max_questions=NUM_GPQA_TRAIN + NUM_GPQA_EVAL,
    )
    gpqa_ids = [q.id for q in gpqa_questions]

    available_ids = [
        qid for qid in gpqa_ids if (forcing_dir / qid).exists()
    ]
    skipped = len(gpqa_ids) - len(available_ids)

    # Compute mean entropy for all available GPQA questions
    all_candidate_ids = (
        CUSTOM_TRAIN_IDS
        + available_ids
        + [eid for eid in EXTRA_EVAL_IDS if (forcing_dir / eid).exists()]
    )
    entropy_map = _compute_question_mean_entropy(forcing_dir, all_candidate_ids)

    # Total questions to distribute across train/val/eval
    n_total = NUM_GPQA_TRAIN + NUM_GPQA_EVAL
    n_available = len(available_ids)
    n_eval = min(NUM_GPQA_EVAL, n_available)
    n_train_val = min(NUM_GPQA_TRAIN, n_available - n_eval)
    n_val = max(1, int(n_train_val * VAL_SPLIT))
    n_train = n_train_val - n_val

    # Stratified split: sort GPQA by entropy, then in each group of
    # (n_train + n_val + n_eval) / n_val ≈ 6 consecutive questions,
    # assign 1 to val, 1 to eval, rest to train. This ensures all
    # splits cover the full entropy range proportionally.
    gpqa_sorted = sorted(available_ids[:n_train_val + n_eval],
                         key=lambda qid: entropy_map.get(qid, 0.0))

    group_size = max(1, len(gpqa_sorted) // max(n_val, 1))  # e.g. 60/10 = 6
    gpqa_train, gpqa_val, gpqa_eval = [], [], []
    for i, qid in enumerate(gpqa_sorted):
        pos_in_group = i % group_size
        if pos_in_group == 0 and len(gpqa_val) < n_val:
            gpqa_val.append(qid)
        elif pos_in_group == 1 and len(gpqa_eval) < n_eval:
            gpqa_eval.append(qid)
        else:
            gpqa_train.append(qid)

    # Custom always train, extra eval always eval
    train_ids = CUSTOM_TRAIN_IDS + gpqa_train
    val_ids = gpqa_val
    eval_ids = gpqa_eval + [
        eid for eid in EXTRA_EVAL_IDS if (forcing_dir / eid).exists()
    ]

    # Shuffle within splits
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(eval_ids)

    # Print entropy balance
    for label, ids in [("Train", train_ids), ("Val", val_ids), ("Eval", eval_ids)]:
        ents = [entropy_map.get(q, 0.0) for q in ids]
        if ents:
            print(f"  {label:5s}: {len(ids)} questions, mean_entropy={np.mean(ents):.4f} (std={np.std(ents):.4f})")

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "eval_ids": eval_ids,
        "n_gpqa_available": n_available,
        "n_gpqa_skipped": skipped,
        "entropy_map": entropy_map,
    }


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # ── Step 1: Build question splits ─────────────────────────────────
    print("Building question splits (stratified by entropy)...")
    splits = build_question_splits(forcing.forcing_dir)
    train_ids = splits["train_ids"]
    val_ids = splits["val_ids"]
    eval_ids = splits["eval_ids"]

    print(f"  GPQA Diamond available on disk: {splits['n_gpqa_available']}")
    if splits["n_gpqa_skipped"] > 0:
        print(f"  GPQA Diamond not yet forced (skipped): {splits['n_gpqa_skipped']}")

    all_question_ids = train_ids + val_ids + eval_ids

    # ── Step 2: Sample sentence indices ───────────────────────────────
    print("\nSampling sentence indices...")
    train_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, train_ids, MAX_SENTENCES_PER_QUESTION_TRAIN, SEED,
    )
    val_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, val_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    eval_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, eval_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    sentence_map = {**train_sentence_map, **val_sentence_map, **eval_sentence_map}
    train_sents = sum(len(v) for v in train_sentence_map.values())
    val_sents = sum(len(v) for v in val_sentence_map.values())
    eval_sents = sum(len(v) for v in eval_sentence_map.values())
    print(f"  Train: {train_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_TRAIN}/question)")
    print(f"  Val:   {val_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_EVAL}/question)")
    print(f"  Eval:  {eval_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_EVAL}/question)")

    # ── Step 3: Extract activations if needed ─────────────────────────
    if EXTRACT_ACTIVATIONS:
        ds = DataSlice(ids=set(all_question_ids))
        print(f"\nExtracting activations for {len(all_question_ids)} questions "
              f"(all sentences per rollout, batched)...")
        forcing.extract_activations_batched(
            model_name=ACTIVATION_MODEL,
            layer=LAYER,
            data_slice=ds,
        )

    # ── Step 4: Load data ───────────────────────────────────────────────
    actual_train_ids = train_ids  # train/val/eval already split in build_question_splits
    tokenizer = None
    if TRIM_TO_COT:
        from transformers import AutoTokenizer
        print("\nLoading tokenizer for CoT trimming...")
        tokenizer = AutoTokenizer.from_pretrained(ACTIVATION_MODEL, trust_remote_code=True)

    print("\nLoading training data...")
    train_data = load_probe_data(
        forcing, actual_train_ids, sentence_map, LAYER, TOKEN_POSITION,
        tokenizer=tokenizer, trim_to_cot=TRIM_TO_COT,
    )
    print(f"  Train: {len(train_data['X_list'])} samples from {len(actual_train_ids)} questions")
    if TRIM_TO_COT and train_data["X_list"]:
        seq_lens = [x.shape[0] for x in train_data["X_list"]]
        print(f"    CoT-only seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    print("Loading validation data...")
    val_data = load_probe_data(
        forcing, val_ids, sentence_map, LAYER, TOKEN_POSITION,
        tokenizer=tokenizer, trim_to_cot=TRIM_TO_COT,
    )
    print(f"  Val:   {len(val_data['X_list'])} samples from {len(val_ids)} questions")

    eval_data = None
    if eval_ids:
        print("Loading eval data...")
        eval_data = load_probe_data(
            forcing, eval_ids, sentence_map, LAYER, TOKEN_POSITION,
            tokenizer=tokenizer, trim_to_cot=TRIM_TO_COT,
        )
        print(f"  Eval:  {len(eval_data['X_list'])} samples from {len(eval_ids)} questions")

    if len(train_data["X_list"]) < 5:
        print("Too few training samples. Exiting.")
        return

    # Print entropy distribution stats
    print(f"\n  Entropy distribution:")
    print(f"    Train: mean={train_data['y_entropy'].mean():.4f}, std={train_data['y_entropy'].std():.4f}, "
          f"min={train_data['y_entropy'].min():.4f}, max={train_data['y_entropy'].max():.4f}")
    print(f"    Val:   mean={val_data['y_entropy'].mean():.4f}, std={val_data['y_entropy'].std():.4f}")
    if eval_data:
        print(f"    Eval:  mean={eval_data['y_entropy'].mean():.4f}, std={eval_data['y_entropy'].std():.4f}")
    # Max possible entropy for 4 classes: log(4) ≈ 1.386
    print(f"    (max possible for 4 classes: {np.log(4):.4f})")

    # ── Step 5b: Per-question mean subtraction ────────────────────────
    if MEAN_SUBTRACT:
        print("\nApplying per-question mean subtraction...")
        train_data = mean_subtract_per_question(train_data)
        val_data = mean_subtract_per_question(val_data)
        if eval_data:
            eval_data = mean_subtract_per_question(eval_data)
        print("  Done — question identity signal removed from activations")

    # ── Step 5c: PCA dimensionality reduction ─────────────────────────
    if USE_PCA:
        print(f"\nApplying PCA: {train_data['X_list'][0].shape[1]} → {BOTTLENECK_DIM}")
        pca = fit_pca(train_data, n_components=BOTTLENECK_DIM)
        train_data = apply_pca(train_data, pca)
        val_data = apply_pca(val_data, pca)
        if eval_data:
            eval_data = apply_pca(eval_data, pca)

    # ── Step 6: Train and evaluate ────────────────────────────────────
    print("\nTraining entropy attention probe...")
    results = train_and_evaluate(
        train_X=train_data["X_list"],
        train_y=train_data["y_entropy"],
        val_X=val_data["X_list"],
        val_y=val_data["y_entropy"],
        test_X=eval_data["X_list"] if eval_data else [],
        test_y=eval_data["y_entropy"] if eval_data else np.zeros(0),
    )

    # ── Step 7: Compute and print metrics ─────────────────────────────
    train_metrics = compute_metrics(
        train_data["y_entropy"],
        results["train_preds"],
        train_data["question_ids"],
    )
    print_results(train_metrics, label="Train Set Results")

    val_metrics = compute_metrics(
        val_data["y_entropy"],
        results["val_preds"],
        val_data["question_ids"],
    )
    print_results(val_metrics, label="Validation Set Results")

    eval_metrics = {}
    if eval_data and len(eval_data["X_list"]) > 0:
        eval_metrics = compute_metrics(
            eval_data["y_entropy"],
            results["test_preds"],
            eval_data["question_ids"],
        )
        print_results(eval_metrics, label="Eval Set Results (held-out questions)")

    # ── Step 8: Save results ──────────────────────────────────────────
    from datetime import datetime
    base_dir = DATA_DIR / "entropy_probe"
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update "latest" symlink
    latest_link = base_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(timestamp)

    # Save key params for quick reference
    params = {
        "bottleneck_dim": BOTTLENECK_DIM,
        "use_linear_baseline": USE_LINEAR_BASELINE,
        "freeze_projection": FREEZE_PROJECTION,
        "use_pca": USE_PCA,
        "mean_subtract": MEAN_SUBTRACT,
        "num_heads": NUM_HEADS,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "grad_clip": GRAD_CLIP,
        "patience": PATIENCE,
        "layer": LAYER,
        "max_sentences_train": MAX_SENTENCES_PER_QUESTION_TRAIN,
        "max_sentences_eval": MAX_SENTENCES_PER_QUESTION_EVAL,
        "n_train": len(train_data["X_list"]),
        "n_val": len(val_data["X_list"]),
        "n_eval": len(eval_data["X_list"]) if eval_data else 0,
        "best_val_loss": results["best_val_loss"],
        "best_epoch": results["best_epoch"],
        "total_epochs": results["total_epochs"],
    }
    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    output = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "eval_metrics": eval_metrics,
        "config": {
            "subject_model": SUBJECT_MODEL,
            "activation_model": ACTIVATION_MODEL,
            "layer": LAYER,
            "token_position": TOKEN_POSITION,
            "trim_to_cot": TRIM_TO_COT,
            "train_question_ids": actual_train_ids,
            "val_question_ids": val_ids,
            "eval_question_ids": eval_ids,
            "max_sentences_per_question_train": MAX_SENTENCES_PER_QUESTION_TRAIN,
            "max_sentences_per_question_eval": MAX_SENTENCES_PER_QUESTION_EVAL,
            "num_gpqa_train": NUM_GPQA_TRAIN,
            "num_gpqa_eval": NUM_GPQA_EVAL,
            "bottleneck_dim": BOTTLENECK_DIM,
            "use_pca": USE_PCA,
            "freeze_projection": FREEZE_PROJECTION,
            "mean_subtract": MEAN_SUBTRACT,
            "num_heads": NUM_HEADS,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "n_train_samples": len(train_data["X_list"]),
            "n_val_samples": len(val_data["X_list"]),
            "n_eval_samples": len(eval_data["X_list"]) if eval_data else 0,
            "sentence_map": {
                qid: [int(i) for i in sorted(indices)] for qid, indices in sentence_map.items()
            },
        },
        "best_val_loss": results["best_val_loss"],
        "best_epoch": results["best_epoch"],
        "total_epochs": results["total_epochs"],
        "final_train_loss": results["final_train_loss"],
        "history": results.get("history", {}),
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    # ── Step 9: Plot training curves ─────────────────────────────────
    if results.get("history") and results["history"]["epoch"]:
        from src2.utils.plotting import plot_training_curves
        plot_training_curves(
            results["history"],
            metric_name="r2",
            output_path=output_dir / "training_curves.png",
            title="Entropy Probe: R² vs Epoch",
        )


if __name__ == "__main__":
    main()
