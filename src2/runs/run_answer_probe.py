#!/usr/bin/env python3
"""
Soft A/B/C/D attention probe.

Trains a generalizable attention probe that predicts soft answer probability
distributions (A/B/C/D) from model activations at forcing boundaries.

Uses soft cross-entropy loss against the actual answer_distribution from
forcing, rather than hard argmax labels.

To avoid over-representing questions with long CoTs, randomly samples a
fixed number of sentence indices per question/rollout for activation
extraction and training.

Usage:
    python -m src2.runs.run_answer_probe
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Sentence sampling: sample this many sentence indices per question.
# Each index appears in all rollouts, so total samples per question ≈
# max_sentences × num_rollouts.
# Keep train low to reduce correlated samples and overfitting.
# Eval can be larger for more reliable metrics.
MAX_SENTENCES_PER_QUESTION_TRAIN = 5   # ~50 samples/question (× 10 rollouts)
MAX_SENTENCES_PER_QUESTION_EVAL = 20   # ~200 samples/question (× 10 rollouts)
ANSWER_LABELS = ["A", "B", "C", "D"]
NUM_CLASSES = 4

# Training hyperparameters
BOTTLENECK_DIM = 64  # project 5120→64 before probe
FREEZE_PROJECTION = True  # frozen random projection (can't memorize) vs learned bottleneck
NUM_HEADS = 2
LR = 4e-4
EPOCHS = 200
BATCH_SIZE = 256
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
SEED = 42

# Validation / early stopping
VAL_SPLIT = 0.2   # fraction of train questions held out for validation
PATIENCE = 20      # stop if val loss doesn't improve for this many epochs
MIN_DELTA = 0.01   # minimum val loss decrease to count as improvement

EXTRACT_ACTIVATIONS = False
TOKEN_POSITION = "full_sequence"  # after re-extraction, can use "cot_only" directly
TRIM_TO_COT = True  # tokenizer-based trim; not needed if TOKEN_POSITION="cot_only"


# ── Helpers ───────────────────────────────────────────────────────────


def sample_sentence_indices(
    forcing_dir: Path,
    question_ids: list,
    max_per_question: int,
    seed: int,
) -> dict:
    """For each question, find available sentence indices and randomly sample N.

    Returns:
        {qid: set(int)} mapping question IDs to sampled sentence indices.
    """
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


def dist_dict_to_array(dist_dict: dict) -> np.ndarray:
    """Convert {"A": 0.4, "B": 0.28, ...} → np.array([0.4, 0.28, 0.13, 0.19]).

    Only includes A/B/C/D keys. Re-normalizes to sum to 1.
    Returns None if no ABCD keys are present.
    """
    vals = [dist_dict.get(label, 0.0) for label in ANSWER_LABELS]
    total = sum(vals)
    if total == 0:
        return None
    return np.array([v / total for v in vals], dtype=np.float32)


def _compute_base_prompt_tokens(full_prompt: str, tokenizer) -> int:
    """Find number of tokens in the base prompt (everything up to and including <think>).

    Returns 0 if <think> not found (no trimming).
    """
    think_tag = "<think>"
    pos = full_prompt.find(think_tag)
    if pos < 0:
        return 0
    # Include the <think> tag and any trailing newline
    end = pos + len(think_tag)
    if end < len(full_prompt) and full_prompt[end] == "\n":
        end += 1
    base_text = full_prompt[:end]
    # Match tokenization used during extraction
    ids = tokenizer(base_text, truncation=True, max_length=4096)["input_ids"]
    return len(ids)


def load_probe_data(
    forcing: ForcingTask,
    question_ids: list,
    sentence_map: dict,
    layer: int,
    token_position: str,
    tokenizer=None,
    trim_to_cot: bool = False,
) -> dict:
    """Load activations + soft distributions for the given questions.

    Uses per-question DataSlice to respect sampled sentences.

    Args:
        tokenizer: HuggingFace tokenizer, required if trim_to_cot=True.
        trim_to_cot: If True, slice full_sequence activations to only include
            tokens after <think> (removes system/user prompt tokens).

    Returns:
        {
            "X_list": list of ndarrays,
            "y_soft": ndarray [n, 4],
            "question_ids": list,
            "sentence_indices": list,
        }
    """
    X_list = []
    y_soft_list = []
    q_ids = []
    s_indices = []

    # Cache base prompt token count per question (same for all sentences)
    base_token_cache = {}

    for qid in question_ids:
        ds = DataSlice(ids={qid}, sentence_indices=sentence_map.get(qid))
        samples = forcing.get_probe_data(layer, ds, token_position)

        for sample in samples:
            dist = dist_dict_to_array(sample["answer_distribution"])
            if dist is None:
                continue

            act = sample["activation"]

            # Trim to CoT-only tokens (remove prompt prefix)
            if trim_to_cot and tokenizer is not None and act.ndim == 2:
                if qid not in base_token_cache:
                    base_token_cache[qid] = _compute_base_prompt_tokens(
                        sample["full_prompt"], tokenizer,
                    )
                n_trim = base_token_cache[qid]
                if n_trim > 0 and act.shape[0] > n_trim:
                    act = act[n_trim:]

            X_list.append(act)
            y_soft_list.append(dist)
            q_ids.append(sample["question_id"])
            s_indices.append(sample["sentence_idx"])

    y_soft = np.stack(y_soft_list) if y_soft_list else np.zeros((0, NUM_CLASSES))

    return {
        "X_list": X_list,
        "y_soft": y_soft,
        "question_ids": q_ids,
        "sentence_indices": s_indices,
    }


def soft_cross_entropy(
    logits: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy: -(target * log_softmax(logits)).sum(-1).mean()"""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


# ── Probe wrappers ────────────────────────────────────────────────


class FrozenProjectionProbe(nn.Module):
    """Fixed random projection → attention pooling → classifier.

    The projection is a frozen random matrix (not learned), so the probe
    cannot memorize question-specific directions. Only the small attention
    probe + classifier are trainable (~12K params).
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        num_heads: int,
        output_dim: int,
        max_seq_len: int,
        dropout: float,
        seed: int = 42,
    ):
        super().__init__()
        # Frozen random projection (JL-style, preserves distances)
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(input_dim, proj_dim, generator=gen) / (proj_dim ** 0.5)
        self.register_buffer("proj_weight", W)

        self.norm = nn.LayerNorm(proj_dim)
        self.attn_probe = AttentionPoolingProbe(
            hidden_dim=proj_dim,
            num_heads=num_heads,
            output_dim=output_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        x = x @ self.proj_weight  # frozen, no gradient
        x = self.norm(x)
        return self.attn_probe(x, mask)


class BottleneckAttentionProbe(nn.Module):
    """Learned projection → attention pooling (for comparison).

    5120 → bottleneck_dim → AttentionPoolingProbe(hidden_dim=bottleneck_dim)
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        num_heads: int,
        output_dim: int,
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
            output_dim=output_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        x = self.proj(x)
        return self.attn_probe(x, mask)


# ── Training ──────────────────────────────────────────────────────────


def _eval_loss_and_acc(probe, X_list, y_soft, device, batch_size=64):
    """Compute soft CE loss and argmax accuracy over a dataset without gradients.

    Returns (loss, argmax_accuracy).
    """
    probe.eval()
    hidden_dim = X_list[0].shape[1]
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for start in range(0, len(X_list), batch_size):
        end = min(start + batch_size, len(X_list))
        batch_X = X_list[start:end]
        batch_y = y_soft[start:end]
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
            loss = soft_cross_entropy(pred, y_t)
        total_loss += loss.item()
        n_batches += 1

        # Argmax accuracy: does probe's top prediction match the true top label?
        pred_labels = pred.argmax(dim=-1)
        true_labels = y_t.argmax(dim=-1)
        correct += (pred_labels == true_labels).sum().item()
        total += len(batch_X)

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def _predict_dists(probe, X_list, device):
    """Get predicted probability distributions for a list of activations."""
    probe.eval()
    pred_dists = []
    for x in X_list:
        seq_len = x.shape[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        m = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = probe(x_t, m)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_dists.append(probs.cpu().numpy())
    return np.array(pred_dists) if pred_dists else np.zeros((0, NUM_CLASSES))


def train_and_evaluate(
    train_X: list,
    train_y_soft: np.ndarray,
    val_X: list,
    val_y_soft: np.ndarray,
    test_X: list,
    test_y_soft: np.ndarray,
    num_heads: int = NUM_HEADS,
    lr: float = LR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train probe with soft CE loss, early stopping on validation set.

    Returns dict with predictions, true labels, and training info.
    """
    import copy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}, batch_size: {batch_size}")

    hidden_dim = train_X[0].shape[1]
    all_X = list(train_X) + list(val_X) + list(test_X)
    max_seq_len = max(x.shape[0] for x in all_X) if all_X else 1
    n_samples = len(train_X)

    if FREEZE_PROJECTION:
        probe = FrozenProjectionProbe(
            input_dim=hidden_dim,
            proj_dim=BOTTLENECK_DIM,
            num_heads=num_heads,
            output_dim=NUM_CLASSES,
            max_seq_len=max_seq_len,
            dropout=DROPOUT,
            seed=SEED,
        ).to(device)
    else:
        probe = BottleneckAttentionProbe(
            input_dim=hidden_dim,
            bottleneck_dim=BOTTLENECK_DIM,
            num_heads=num_heads,
            output_dim=NUM_CLASSES,
            max_seq_len=max_seq_len,
            dropout=DROPOUT,
        ).to(device)
    n_trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in probe.parameters())
    proj_label = "frozen" if FREEZE_PROJECTION else "learned"
    print(f"  Probe: {n_trainable:,} trainable / {n_total:,} total params ({proj_label} projection {hidden_dim}→{BOTTLENECK_DIM})")
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # Use ReduceLROnPlateau so schedule responds to actual val loss, not a fixed cycle
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

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
            batch_y = train_y_soft[batch_idx]
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
            loss = soft_cross_entropy(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        if val_X:
            val_loss, val_acc = _eval_loss_and_acc(probe, val_X, val_y_soft, device)
        else:
            val_loss, val_acc = train_loss, 0.0

        scheduler.step(val_loss)

        # ── Early stopping ──
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(probe.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            # Only compute train acc on log epochs to save compute
            _, train_acc = _eval_loss_and_acc(probe, train_X, train_y_soft, device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}/{epochs}  "
                f"train: {train_loss:.4f} (acc {train_acc:.3f})  "
                f"val: {val_loss:.4f} (acc {val_acc:.3f})  "
                f"best_val: {best_val_loss:.4f} (ep {best_epoch})  "
                f"lr: {current_lr:.2e}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1} (no val improvement for {PATIENCE} epochs)")
            break

    # Restore best model
    if best_state is not None:
        probe.load_state_dict(best_state)
        print(f"  Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # ── Predictions ──
    return {
        "test_pred_dists": _predict_dists(probe, test_X, device),
        "train_pred_dists": _predict_dists(probe, train_X, device),
        "val_pred_dists": _predict_dists(probe, val_X, device),
        "final_train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
    }


# ── Metrics ───────────────────────────────────────────────────────────


def compute_metrics(
    y_true_soft: np.ndarray,
    y_pred_soft: np.ndarray,
    question_ids: list,
) -> dict:
    """Compute hard accuracy, mean KL divergence, and per-question breakdown.

    Args:
        y_true_soft: [n, 4] true soft distributions
        y_pred_soft: [n, 4] predicted soft distributions
        question_ids: list of question IDs for per-question breakdown
    """
    n = len(y_true_soft)
    if n == 0:
        return {"n_samples": 0}

    # Hard accuracy (argmax match)
    true_hard = y_true_soft.argmax(axis=1)
    pred_hard = y_pred_soft.argmax(axis=1)
    hard_accuracy = (true_hard == pred_hard).mean()

    # Chance baseline
    class_counts = Counter(true_hard.tolist())
    chance_baseline = max(class_counts.values()) / n if n > 0 else 0.0

    # Soft cross-entropy: -sum(true * log(pred))
    eps = 1e-8
    pred_clipped = np.clip(y_pred_soft, eps, 1.0)
    true_clipped = np.clip(y_true_soft, eps, 1.0)
    soft_ce_per_sample = -(true_clipped * np.log(pred_clipped)).sum(axis=1)
    mean_soft_ce = float(soft_ce_per_sample.mean())

    # Mean KL divergence: KL(true || pred)
    kl_per_sample = (true_clipped * np.log(true_clipped / pred_clipped)).sum(axis=1)
    mean_kl = float(kl_per_sample.mean())

    # Confusion matrix (hard labels): rows = true, cols = predicted
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(true_hard, pred_hard):
        conf_matrix[t, p] += 1

    # Per-question breakdown
    per_question = {}
    unique_qids = sorted(set(question_ids))
    for qid in unique_qids:
        idx = [i for i, q in enumerate(question_ids) if q == qid]
        q_true = true_hard[idx]
        q_pred = pred_hard[idx]
        q_kl = kl_per_sample[idx]
        q_ce = soft_ce_per_sample[idx]
        per_question[qid] = {
            "n_samples": len(idx),
            "hard_accuracy": float((q_true == q_pred).mean()),
            "mean_soft_ce": float(q_ce.mean()),
            "mean_kl": float(q_kl.mean()),
        }

    return {
        "hard_accuracy": float(hard_accuracy),
        "chance_baseline": float(chance_baseline),
        "mean_soft_ce": mean_soft_ce,
        "mean_kl_divergence": mean_kl,
        "n_samples": n,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_labels": ANSWER_LABELS,
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

    print(f"  Argmax accuracy: {metrics['hard_accuracy']:.3f}  (does top predicted label match top true label?)")
    print(f"  Chance baseline: {metrics['chance_baseline']:.3f}")
    print(f"  Soft CE loss:    {metrics['mean_soft_ce']:.4f}  (distribution-level match)")
    print(f"  Mean KL div:     {metrics['mean_kl_divergence']:.4f}")
    print(f"  N samples:       {metrics['n_samples']}")

    if "per_question" in metrics:
        print(f"\n  Per-question breakdown:")
        for qid, vals in metrics["per_question"].items():
            print(
                f"    {qid:>25s}: "
                f"argmax_acc={vals['hard_accuracy']:.3f}  "
                f"soft_ce={vals['mean_soft_ce']:.4f}  "
                f"KL={vals['mean_kl']:.4f}  "
                f"n={vals['n_samples']}"
            )

    if "confusion_matrix" in metrics:
        print(f"\n  Confusion matrix (rows=true, cols=predicted):")
        labels = metrics["confusion_labels"]
        header = "            " + "  ".join(f"{l:>6s}" for l in labels)
        print(header)
        for i, row in enumerate(metrics["confusion_matrix"]):
            row_str = "  ".join(f"{v:>6d}" for v in row)
            print(f"    {labels[i]:>6s}  {row_str}")


# ── Main ──────────────────────────────────────────────────────────────


def build_question_splits(forcing_dir: Path, seed: int = SEED) -> dict:
    """Build train/eval question ID lists from custom + GPQA Diamond questions.

    Loads GPQA Diamond from HuggingFace, deterministically shuffles, and
    splits into train/eval. Only includes questions that have forcing data
    on disk (skips questions that haven't been forced yet).

    Returns:
        {"train_ids": list[str], "eval_ids": list[str],
         "n_gpqa_available": int, "n_gpqa_skipped": int}
    """
    rng = np.random.default_rng(seed)

    # Load all GPQA Diamond questions
    gpqa_questions = load_gpqa_from_huggingface(
        subset="gpqa_diamond", max_questions=NUM_GPQA_TRAIN + NUM_GPQA_EVAL,
    )
    gpqa_ids = [q.id for q in gpqa_questions]

    # Filter to only questions with forcing data on disk
    available_ids = [
        qid for qid in gpqa_ids if (forcing_dir / qid).exists()
    ]
    skipped = len(gpqa_ids) - len(available_ids)

    # Deterministic shuffle then split
    available_ids = list(available_ids)
    rng.shuffle(available_ids)
    gpqa_train = available_ids[:NUM_GPQA_TRAIN]
    gpqa_eval = available_ids[NUM_GPQA_TRAIN : NUM_GPQA_TRAIN + NUM_GPQA_EVAL]

    train_ids = CUSTOM_TRAIN_IDS + gpqa_train
    eval_ids = gpqa_eval + [
        eid for eid in EXTRA_EVAL_IDS if (forcing_dir / eid).exists()
    ]

    return {
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "n_gpqa_available": len(available_ids),
        "n_gpqa_skipped": skipped,
    }


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # ── Step 1: Build question splits ─────────────────────────────────
    print("Building question splits from custom + GPQA Diamond...")
    splits = build_question_splits(forcing.forcing_dir)
    train_ids = splits["train_ids"]
    eval_ids = splits["eval_ids"]

    print(f"  GPQA Diamond available on disk: {splits['n_gpqa_available']}")
    if splits["n_gpqa_skipped"] > 0:
        print(f"  GPQA Diamond not yet forced (skipped): {splits['n_gpqa_skipped']}")
    print(f"  Train questions: {len(train_ids)} ({len(CUSTOM_TRAIN_IDS)} custom + {len(train_ids) - len(CUSTOM_TRAIN_IDS)} GPQA)")
    print(f"  Eval questions:  {len(eval_ids)}")

    all_question_ids = train_ids + eval_ids

    # ── Step 2: Sample sentence indices (fewer for train, more for eval) ─
    print("\nSampling sentence indices...")
    train_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, train_ids, MAX_SENTENCES_PER_QUESTION_TRAIN, SEED,
    )
    eval_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, eval_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    sentence_map = {**train_sentence_map, **eval_sentence_map}
    train_sents = sum(len(v) for v in train_sentence_map.values())
    eval_sents = sum(len(v) for v in eval_sentence_map.values())
    print(f"  Train/val: {train_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_TRAIN}/question)")
    print(f"  Eval:      {eval_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_EVAL}/question)")

    # ── Step 3: Extract activations if needed ─────────────────────────
    if EXTRACT_ACTIVATIONS:
        # No sentence_indices filter: batched extraction runs one forward pass
        # per rollout covering ALL sentences. Sentence sampling happens at
        # data-loading time (step 4).
        ds = DataSlice(ids=set(all_question_ids))
        print(f"\nExtracting activations for {len(all_question_ids)} questions "
              f"(all sentences per rollout, batched)...")
        forcing.extract_activations_batched(
            model_name=ACTIVATION_MODEL,
            layer=LAYER,
            data_slice=ds,
        )

    # ── Step 4: Split train into train/val by question_id ───────────
    rng = np.random.default_rng(SEED)
    shuffled_train = list(train_ids)
    rng.shuffle(shuffled_train)
    n_val = max(1, int(len(shuffled_train) * VAL_SPLIT))
    val_ids = shuffled_train[:n_val]
    actual_train_ids = shuffled_train[n_val:]
    print(f"\n  Train/val split: {len(actual_train_ids)} train questions, {len(val_ids)} val questions")

    # ── Step 5: Load data ─────────────────────────────────────────────
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

    # ── Step 6: Train and evaluate ────────────────────────────────────
    print("\nTraining soft-label attention probe...")
    results = train_and_evaluate(
        train_X=train_data["X_list"],
        train_y_soft=train_data["y_soft"],
        val_X=val_data["X_list"],
        val_y_soft=val_data["y_soft"],
        test_X=eval_data["X_list"] if eval_data else [],
        test_y_soft=eval_data["y_soft"] if eval_data else np.zeros((0, NUM_CLASSES)),
    )

    # ── Step 7: Compute and print metrics ─────────────────────────────
    train_metrics = compute_metrics(
        train_data["y_soft"],
        results["train_pred_dists"],
        train_data["question_ids"],
    )
    print_results(train_metrics, label="Train Set Results")

    val_metrics = compute_metrics(
        val_data["y_soft"],
        results["val_pred_dists"],
        val_data["question_ids"],
    )
    print_results(val_metrics, label="Validation Set Results")

    eval_metrics = {}
    if eval_data and len(eval_data["X_list"]) > 0:
        eval_metrics = compute_metrics(
            eval_data["y_soft"],
            results["test_pred_dists"],
            eval_data["question_ids"],
        )
        print_results(eval_metrics, label="Eval Set Results (held-out questions)")

    # ── Step 8: Save results ──────────────────────────────────────────
    output_dir = DATA_DIR / "answer_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "freeze_projection": FREEZE_PROJECTION,
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


if __name__ == "__main__":
    main()
