"""
Run HeatmapCNN for majority/minority answer classification
using leave-one-out (LOO) cross-validation across all 7 prompts.

Four phases:
  1. Data loading — build rollout entries from MinMajAnswerTask
  2. Activation extraction — extract layer-44 activations (GPU, skippable)
  3. Heatmap image building — build 3-channel KxK images once for all folds
  4. LOO-CV — train/eval fresh HeatmapCNN per fold, aggregate metrics

Usage:
    # Full run (on GPU):
    python -m src2.runs.run_min_maj_heatmap_cnn

    # Re-run CV with different hyperparams (skip extraction):
    python -m src2.runs.run_min_maj_heatmap_cnn --skip-extraction --epochs 120 --K 40
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src2.methods.heatmap_cnn import (
    HeatmapCNN,
    HeatmapDataset,
    _build_sample,
    build_heatmap_image,
    cosine_sim_matrix,
    centered_cosine_sim_matrix,
    l2_distance_sim_matrix,
    build_sentence_segments,
)
from src2.tasks import MinMajAnswerTask
from src2.tasks.min_maj_answer.task import ALL_PROMPT_IDS
from src2.utils.activations import ActivationExtractor

# ── Constants ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
ACT_DIR = DATA_DIR / "activations_layer44" / "layer44"
LOO_DIR = DATA_DIR / "loo_heatmap_cnn"

MODEL_NAME = "Qwen/Qwen3-32B"
LABEL_MAP = {0: "minority", 1: "majority"}
LABEL_TO_INT = {"minority": 0, "majority": 1}


# ── Phase 1: Data Loading ───────────────────────────────────────────

def build_entries(task: MinMajAnswerTask) -> list:
    """Build entry dicts from the rollout DataFrame."""
    df = task._build_rollout_df(ALL_PROMPT_IDS)
    entries = []
    for _, row in df.iterrows():
        messages = [
            {"role": "user", "content": row["prompt_text"]},
            {"role": "assistant", "content": row["cot_content"]},
        ]
        entries.append({
            "messages": messages,
            "label": LABEL_TO_INT[row["label"]],
            "label_str": row["label"],
            "prompt_id": row["prompt_id"],
            "prompt_name": row["prompt_id"],
            "rollout_idx": row["rollout_idx"],
            "answer": row["answer"],
        })
    return entries


# ── Phase 2: Activation Extraction ──────────────────────────────────

def extract_activations(entries: list, layer: int, batch_size: int = 8):
    """Extract full-sequence activations in batches using ActivationExtractor."""
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    extractor = ActivationExtractor(
        model_name=MODEL_NAME,
        load_in_4bit=False,
    )

    # Find entries that need extraction
    todo = []
    n_existing = 0
    for idx, entry in enumerate(entries):
        out_path = ACT_DIR / f"sample_{idx}.npy"
        if out_path.exists():
            n_existing += 1
        else:
            todo.append(idx)

    print(f"  {n_existing} cached, {len(todo)} to extract (batch_size={batch_size})")

    if not todo:
        return

    # Prepare all texts upfront
    tokenizer = extractor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad so real tokens are right-aligned
    texts = {}
    for idx in todo:
        texts[idx] = tokenizer.apply_chat_template(
            entries[idx]["messages"], tokenize=False, add_generation_prompt=False,
        )

    # Process in batches
    n_extracted = 0
    for batch_start in range(0, len(todo), batch_size):
        batch_idxs = todo[batch_start:batch_start + batch_size]
        batch_texts = [texts[idx] for idx in batch_idxs]

        # Tokenize with padding
        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=4096, padding=True,
        ).to(extractor.model.device)

        # Extract activations via hook
        activations = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations["resid"] = output[0].detach()
            else:
                activations["resid"] = output.detach()

        handle = extractor.model.model.layers[layer].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                extractor.model(**inputs)
        finally:
            handle.remove()

        # Save per-sample, stripping padding
        attn_mask = inputs["attention_mask"]
        for i, idx in enumerate(batch_idxs):
            seq_len = int(attn_mask[i].sum())
            # Left-padded: take the last seq_len tokens
            acts = activations["resid"][i, -seq_len:, :].cpu().float().numpy()
            np.save(ACT_DIR / f"sample_{idx}.npy", acts)

        n_extracted += len(batch_idxs)
        if n_extracted % 50 < batch_size:
            print(f"  Extracted {n_extracted}/{len(todo)}...")

    print(f"  Activation extraction complete: {n_extracted} new, {n_existing} cached")


# ── Phase 3: Build Heatmap Images ───────────────────────────────────

def build_all_images(entries: list, K: int):
    """Build heatmap images for all entries. Returns list of (image, label, meta) or None per entry."""
    from transformers import AutoTokenizer

    import nltk
    for resource in ["punkt_tab", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    print(f"  Loading tokenizer for image building...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    results = [None] * len(entries)
    n_built = 0
    n_skipped = 0

    for idx, entry in enumerate(entries):
        sample = _build_sample(entry, idx, tokenizer, ACT_DIR, K)
        if sample is None:
            n_skipped += 1
            continue

        img, label, n_sents = sample
        results[idx] = (img, label, {
            "prompt_id": entry["prompt_id"],
            "rollout_idx": entry["rollout_idx"],
            "label_str": entry["label_str"],
            "answer": entry["answer"],
            "n_sents": n_sents,
            "global_idx": idx,
        })
        n_built += 1

        if (idx + 1) % 200 == 0:
            print(f"    {idx + 1}/{len(entries)} processed...")

    n_pos = sum(1 for r in results if r is not None and r[1] == 1)
    print(f"  Image building complete: {n_built} built ({n_pos} majority, "
          f"{n_built - n_pos} minority), {n_skipped} skipped")

    return results, tokenizer


# ── Phase 4: LOO-CV ─────────────────────────────────────────────────

def train_and_eval_fold(
    train_images, train_labels, test_images, test_labels,
    *, epochs, patience, lr, batch_size, seed, dropout=0.5,
    label_smoothing=0.1, noise_std=0.03, weight_decay=1e-3,
    val_fraction=0.15,
):
    """Train a fresh HeatmapCNN on train split, evaluate on test split.

    Returns (test_preds, test_probs, train_info).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Stratified train/val split from training data
    n_val = max(2, int(len(train_images) * val_fraction))
    indices = np.arange(len(train_images))
    labels_arr = np.array(train_labels)

    val_idx = []
    for cls in [0, 1]:
        cls_idx = indices[labels_arr == cls]
        np.random.shuffle(cls_idx)
        n_cls_val = max(1, int(len(cls_idx) * val_fraction))
        val_idx.extend(cls_idx[:n_cls_val].tolist())
    tr_idx = sorted(set(range(len(train_images))) - set(val_idx))
    val_idx = sorted(val_idx)

    tr_imgs = [train_images[i] for i in tr_idx]
    tr_lbls = [train_labels[i] for i in tr_idx]
    v_imgs = [train_images[i] for i in val_idx]
    v_lbls = [train_labels[i] for i in val_idx]

    print(f"    Train: {len(tr_imgs)} ({sum(tr_lbls)} maj), "
          f"Val: {len(v_imgs)} ({sum(v_lbls)} maj)")

    tr_ds = HeatmapDataset(tr_imgs, tr_lbls)
    tr_ds.noise_std = noise_std
    v_ds = HeatmapDataset(v_imgs, v_lbls)
    test_ds = HeatmapDataset(test_images, test_labels)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    v_loader = DataLoader(v_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapCNN(in_channels=3, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_auc = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        val_auc = _eval_auc(model, v_loader, device)
        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or marker:
            print(f"      Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                  f"val_auc={val_auc:.3f}{marker}")

        if patience_counter >= patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break

    # Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().to(device)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    model.cpu()

    train_info = {
        "best_epoch": best_epoch + 1,
        "best_val_auc": best_val_auc,
        "n_train": len(tr_imgs),
        "n_val": len(v_imgs),
    }

    return all_preds, all_probs, train_info


def _eval_auc(model, loader, device):
    """Compute AUC-ROC on a loader."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1]
            all_labels.extend(lbls.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    try:
        return roc_auc_score(all_labels, all_probs)
    except ValueError:
        return 0.0


# ── Visualization ────────────────────────────────────────────────────

def save_sample_heatmaps(entries, all_images, run_dir, tokenizer, n_per_prompt=3):
    """Save a few individual rollout heatmaps per prompt for inspection."""
    viz_dir = run_dir / "visualizations" / "samples"
    viz_dir.mkdir(parents=True, exist_ok=True)

    channel_names = ["cosine", "centered_cosine", "L2"]

    # Group by prompt
    by_prompt = {}
    for idx, result in enumerate(all_images):
        if result is None:
            continue
        img, label, meta = result
        pid = meta["prompt_id"]
        if pid not in by_prompt:
            by_prompt[pid] = []
        by_prompt[pid].append((idx, img, label, meta))

    for pid, items in by_prompt.items():
        selected = items[:n_per_prompt]
        for idx, img, label, meta in selected:
            label_str = meta["label_str"]
            ridx = meta["rollout_idx"]
            # img is (3, K, K) tensor
            img_np = img.numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ch in range(3):
                mat = img_np[ch]
                im = axes[ch].imshow(mat, cmap="RdBu_r", vmin=0, vmax=1,
                                     aspect="equal", interpolation="nearest")
                axes[ch].set_title(f"{channel_names[ch]}", fontsize=11)
                axes[ch].set_xlabel("Sentence index")
                axes[ch].set_ylabel("Sentence index")
                fig.colorbar(im, ax=axes[ch], fraction=0.046, pad=0.04)

            fig.suptitle(f"{pid} / rollout {ridx} — {label_str}", fontsize=13)
            plt.tight_layout()
            out_path = viz_dir / f"{pid}_rollout{ridx}_{label_str}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"  Sample heatmaps saved to {viz_dir}")


def save_average_heatmaps(entries, all_images, run_dir):
    """Compute and save mean heatmap per class per prompt per channel."""
    viz_dir = run_dir / "visualizations" / "averages"
    viz_dir.mkdir(parents=True, exist_ok=True)

    channel_names = ["cosine", "centered_cosine", "L2"]

    # Group images by (prompt_id, label_str)
    groups = {}
    for idx, result in enumerate(all_images):
        if result is None:
            continue
        img, label, meta = result
        key = (meta["prompt_id"], meta["label_str"])
        if key not in groups:
            groups[key] = []
        groups[key].append(img.numpy())

    # Get all prompt ids that have data
    prompt_ids = sorted(set(k[0] for k in groups.keys()))

    for pid in prompt_ids:
        for ch_idx, ch_name in enumerate(channel_names):
            # Get mean images for each class
            class_means = {}
            for label_str in ["majority", "minority"]:
                key = (pid, label_str)
                if key not in groups or len(groups[key]) == 0:
                    continue
                imgs = np.stack([img[ch_idx] for img in groups[key]])
                class_means[label_str] = imgs.mean(axis=0)

                # Save individual average
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(class_means[label_str], cmap="RdBu_r", vmin=0, vmax=1,
                               aspect="equal", interpolation="nearest")
                ax.set_title(f"{pid} — {label_str} avg — {ch_name}\n(n={len(groups[key])})",
                             fontsize=11)
                ax.set_xlabel("Sentence index")
                ax.set_ylabel("Sentence index")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                out_path = viz_dir / f"{pid}_ch{ch_idx}_{label_str}.png"
                fig.savefig(out_path, dpi=120, bbox_inches="tight")
                plt.close(fig)

            # Save side-by-side comparison if both classes present
            if len(class_means) == 2:
                fig, axes = plt.subplots(1, 3, figsize=(20, 5))

                for ax_idx, (label_str, title_prefix) in enumerate(
                    [("majority", "Majority"), ("minority", "Minority")]
                ):
                    n_samples = len(groups[(pid, label_str)])
                    im = axes[ax_idx].imshow(
                        class_means[label_str], cmap="RdBu_r", vmin=0, vmax=1,
                        aspect="equal", interpolation="nearest",
                    )
                    axes[ax_idx].set_title(f"{title_prefix} (n={n_samples})", fontsize=11)
                    axes[ax_idx].set_xlabel("Sentence index")
                    axes[ax_idx].set_ylabel("Sentence index")
                    fig.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

                # Difference plot
                diff = class_means["majority"] - class_means["minority"]
                vabs = max(abs(diff.min()), abs(diff.max()), 0.01)
                im = axes[2].imshow(
                    diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                    aspect="equal", interpolation="nearest",
                )
                axes[2].set_title("Majority − Minority", fontsize=11)
                axes[2].set_xlabel("Sentence index")
                axes[2].set_ylabel("Sentence index")
                fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

                fig.suptitle(f"{pid} — {ch_name} channel", fontsize=13)
                plt.tight_layout()
                out_path = viz_dir / f"{pid}_ch{ch_idx}_comparison.png"
                fig.savefig(out_path, dpi=120, bbox_inches="tight")
                plt.close(fig)

    print(f"  Average heatmaps saved to {viz_dir}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HeatmapCNN LOO-CV for majority/minority answer classification"
    )
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip activation extraction (use cached)")
    parser.add_argument("--layer", type=int, default=44,
                        help="Layer to extract activations from (default: 44)")
    parser.add_argument("--K", type=int, default=30,
                        help="Heatmap size K (number of sentences, default: 30)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Max training epochs (default: 80)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="CNN training batch size (default: 32)")
    parser.add_argument("--extract-batch-size", type=int, default=8,
                        help="Activation extraction batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Update ACT_DIR for the specified layer
    global ACT_DIR
    ACT_DIR = DATA_DIR / f"activations_layer{args.layer}" / f"layer{args.layer}"

    print(f"{'=' * 60}")
    print(f"  HeatmapCNN LOO-CV for Min/Maj Answer Task")
    print(f"{'=' * 60}")
    print(f"  Layer: {args.layer}, K: {args.K}, Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Activations dir: {ACT_DIR}")
    print()

    # ── Phase 1: Data loading ────────────────────────────────────────
    print("Phase 1: Loading data...")
    task = MinMajAnswerTask(
        prompt_ids=ALL_PROMPT_IDS,
        model="qwen3-32b",
        data_dir=DATA_DIR,
    )
    task.run_data()

    entries = build_entries(task)
    print(f"  {len(entries)} rollouts across {len(ALL_PROMPT_IDS)} prompts")

    # Print per-prompt counts
    from collections import Counter
    prompt_counts = Counter(e["prompt_id"] for e in entries)
    for pid in ALL_PROMPT_IDS:
        n = prompt_counts.get(pid, 0)
        n_maj = sum(1 for e in entries if e["prompt_id"] == pid and e["label"] == 1)
        print(f"    {pid}: {n} rollouts ({n_maj} majority, {n - n_maj} minority)")

    # ── Phase 2: Activation extraction ───────────────────────────────
    if args.skip_extraction:
        print("\nPhase 2: Skipping activation extraction (--skip-extraction)")
    else:
        print(f"\nPhase 2: Extracting layer-{args.layer} activations...")
        extract_activations(entries, layer=args.layer, batch_size=args.extract_batch_size)

    # ── Phase 3: Build heatmap images ────────────────────────────────
    print(f"\nPhase 3: Building heatmap images (K={args.K})...")
    all_images, tokenizer = build_all_images(entries, K=args.K)

    # ── Create run directory ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOO_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = LOO_DIR / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(run_dir.name, latest)
    print(f"\nRun directory: {run_dir}")

    # ── Visualizations ───────────────────────────────────────────────
    print("\nGenerating visualizations...")
    save_sample_heatmaps(entries, all_images, run_dir, tokenizer)
    save_average_heatmaps(entries, all_images, run_dir)

    # ── Phase 4: LOO-CV ──────────────────────────────────────────────
    print(f"\nPhase 4: LOO Cross-Validation")
    folds = MinMajAnswerTask.loo_folds()
    print(f"  {len(folds)} folds")

    all_predictions = []
    all_ground_truth = []
    all_fold_results = []
    fold_metrics = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        test_pid = fold["test_prompt_id"]
        train_pids = fold["train_prompt_ids"]

        print(f"\n--- Fold {fold_idx}: test={test_pid}, train={train_pids} ---")

        # Partition pre-built images by prompt_id
        train_imgs, train_lbls = [], []
        test_imgs, test_lbls = [], []
        test_metas = []

        for idx, result in enumerate(all_images):
            if result is None:
                continue
            img, label, meta = result
            if meta["prompt_id"] == test_pid:
                test_imgs.append(img)
                test_lbls.append(label)
                test_metas.append(meta)
            elif meta["prompt_id"] in train_pids:
                train_imgs.append(img)
                train_lbls.append(label)

        if not test_imgs:
            print(f"  No test images for {test_pid}, skipping")
            continue

        print(f"  Train: {len(train_imgs)}, Test: {len(test_imgs)}")

        # Train and evaluate
        preds_int, probs, train_info = train_and_eval_fold(
            train_imgs, train_lbls, test_imgs, test_lbls,
            epochs=args.epochs, patience=args.patience, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed,
        )

        # Convert int predictions to string labels
        preds_str = [LABEL_MAP[p] for p in preds_int]
        gt_str = [LABEL_MAP[l] for l in test_lbls]

        # Evaluate using task.evaluate for consistency
        metrics = task.evaluate(preds_str, gt_str)
        print(f"  Fold {fold_idx}: F1={metrics['macro_f1']:.3f}, "
              f"acc={metrics['accuracy']:.3f} (n={metrics['n_total']})")
        print(f"    best_epoch={train_info['best_epoch']}, "
              f"val_auc={train_info['best_val_auc']:.3f}")

        fold_metrics.append({
            "fold_idx": fold_idx,
            "test_prompt_id": test_pid,
            **metrics,
            **train_info,
        })
        all_predictions.extend(preds_str)
        all_ground_truth.extend(gt_str)

        for i, meta in enumerate(test_metas):
            all_fold_results.append({
                "prompt_id": meta["prompt_id"],
                "rollout_idx": meta["rollout_idx"],
                "answer": meta["answer"],
                "label": gt_str[i],
                "prediction": preds_str[i],
                "prob_majority": float(probs[i]),
                "correct": preds_str[i] == gt_str[i],
            })

    # ── Aggregate metrics ────────────────────────────────────────────
    pooled_metrics = task.evaluate(all_predictions, all_ground_truth)
    fold_f1s = [fm["macro_f1"] for fm in fold_metrics]
    mean_fold_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0

    print(f"\n{'=' * 60}")
    print(f"  LOO Aggregate Results ({len(fold_metrics)} folds)")
    print(f"{'=' * 60}")
    print(f"  Pooled macro F1:    {pooled_metrics['macro_f1']:.3f} "
          f"(n={pooled_metrics['n_total']})")
    print(f"  Mean fold F1:       {mean_fold_f1:.3f}")
    print(f"  Pooled accuracy:    {pooled_metrics['accuracy']:.3f}")
    print(f"  Majority F1:        {pooled_metrics['majority_f1']:.3f}")
    print(f"  Minority F1:        {pooled_metrics['minority_f1']:.3f}")

    print(f"\n  Per-fold breakdown:")
    for fm in fold_metrics:
        print(f"    Fold {fm['fold_idx']} ({fm['test_prompt_id']}): "
              f"F1={fm['macro_f1']:.3f}, acc={fm['accuracy']:.3f}, "
              f"n={fm['n_total']}, best_epoch={fm['best_epoch']}")

    # ── Save results ─────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(run_dir / "loo_fold_metrics.csv", index=False)

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(run_dir / "loo_results.csv", index=False)

    run_config = {
        "method": "heatmap_cnn",
        "model": MODEL_NAME,
        "layer": args.layer,
        "K": args.K,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_folds": len(fold_metrics),
        "n_entries": len(entries),
        "pooled_macro_f1": pooled_metrics["macro_f1"],
        "mean_fold_f1": mean_fold_f1,
        "pooled_accuracy": pooled_metrics["accuracy"],
        "pooled_metrics": pooled_metrics,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    print(f"  Fold metrics:   loo_fold_metrics.csv")
    print(f"  Full results:   loo_results.csv")
    print(f"  Run config:     run_config.json")
    print(f"  Visualizations: visualizations/")


if __name__ == "__main__":
    main()
