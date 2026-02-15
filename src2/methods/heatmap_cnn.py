"""
Heatmap CNN method.

Builds 3-channel KxK images from pairwise similarity matrices of sentence-level
activation means (using a single layer with three different similarity metrics),
then trains a small CNN classifier.

Channels:
  - Channel 0: Cosine similarity
  - Channel 1: Centered cosine similarity (subtract per-sample mean vector)
  - Channel 2: L2 distance similarity (1 / (1 + ||a_i - a_j||))

Early stopping uses a held-out validation split (from training data),
NOT the eval/test set.

Expected data formats:
  train: {"entries": list of dicts with "messages" and "label",
          "activations_dir": Path to per-sample .npy files,
          "tokenizer": HF tokenizer}
  infer: same format
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import BaseMethod


# ── Similarity Matrices ──────────────────────────────────────────────

def cosine_sim_matrix(vectors):
    """Cosine similarity matrix, rescaled to [0, 1]."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = vectors / norms
    sim = normed @ normed.T
    return (sim + 1.0) / 2.0


def centered_cosine_sim_matrix(vectors):
    """Centered cosine similarity (subtract mean vector first), rescaled to [0, 1]."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = centered / norms
    sim = normed @ normed.T
    return (sim + 1.0) / 2.0


def l2_distance_sim_matrix(vectors):
    """L2 distance similarity: 1 / (1 + ||a_i - a_j||), values in (0, 1]."""
    norms_sq = (vectors ** 2).sum(axis=1)
    dists_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * vectors @ vectors.T
    dists_sq = np.maximum(dists_sq, 0.0)
    dists = np.sqrt(dists_sq)
    return 1.0 / (1.0 + dists)


def build_heatmap_image(sentence_vecs, K):
    """Build 3-channel KxK heatmap from sentence activation vectors.

    Pads with 0.5 (neutral) if fewer than K sentences, or takes last K.
    Returns: (3, K, K) float32 tensor.
    """
    N = sentence_vecs.shape[0]

    ch0 = cosine_sim_matrix(sentence_vecs)
    ch1 = centered_cosine_sim_matrix(sentence_vecs)
    ch2 = l2_distance_sim_matrix(sentence_vecs)

    img = np.stack([ch0, ch1, ch2], axis=0)  # (3, N, N)

    if N >= K:
        img = img[:, -K:, -K:]
    else:
        padded = np.full((3, K, K), 0.5, dtype=np.float32)
        padded[:, -N:, -N:] = img
        img = padded

    return torch.from_numpy(img.astype(np.float32))


# ── Token-to-Sentence Mapping ────────────────────────────────────────

def _split_sentences(text):
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    tok = PunktSentenceTokenizer()
    return tok.tokenize(text)


def get_message_token_boundaries(tokenizer, messages):
    """Get token boundaries for each message by progressive tokenization."""
    boundaries = []
    prev_len = 0
    for k in range(1, len(messages) + 1):
        result = tokenizer.apply_chat_template(
            messages[:k], add_generation_prompt=False, tokenize=True,
        )
        token_ids = result["input_ids"] if hasattr(result, "keys") else result
        boundaries.append((prev_len, len(token_ids)))
        prev_len = len(token_ids)
    return boundaries, prev_len


def build_sentence_segments(tokenizer, messages):
    """Build sentence segments with approximate token ranges.

    User messages -> 1 segment each. Assistant messages -> split by nltk punkt.
    """
    msg_boundaries, total_tokens = get_message_token_boundaries(tokenizer, messages)

    segments = []
    for k, msg in enumerate(messages):
        msg_start, msg_end = msg_boundaries[k]
        content = msg["content"]
        role = msg["role"]

        if role == "user":
            segments.append({
                "text": content, "role": role,
                "tok_start": msg_start, "tok_end": msg_end,
            })
        else:
            sentences = _split_sentences(content)
            if not sentences:
                segments.append({
                    "text": content, "role": role,
                    "tok_start": msg_start, "tok_end": msg_end,
                })
                continue

            total_chars = len(content)
            if total_chars == 0:
                continue

            n_msg_tokens = msg_end - msg_start
            char_pos = 0
            for sent in sentences:
                sent_start_char = content.find(sent, char_pos)
                if sent_start_char == -1:
                    sent_start_char = char_pos
                sent_end_char = sent_start_char + len(sent)

                tok_start = msg_start + int(sent_start_char / total_chars * n_msg_tokens)
                tok_end = msg_start + int(sent_end_char / total_chars * n_msg_tokens)
                tok_end = max(tok_end, tok_start + 1)
                tok_end = min(tok_end, msg_end)

                segments.append({
                    "text": sent, "role": role,
                    "tok_start": tok_start, "tok_end": tok_end,
                })
                char_pos = sent_end_char

    return segments, total_tokens


# ── Dataset ──────────────────────────────────────────────────────────

def _build_sample(entry, idx, tokenizer, act_dir, num_sentences):
    """Build a single heatmap image from an entry. Returns (image, label) or None."""
    act_path = act_dir / f"sample_{idx}.npy"
    if not act_path.exists():
        return None

    activations = np.load(act_path)
    if not np.all(np.isfinite(activations)):
        activations = np.nan_to_num(
            activations, nan=0.0,
            posinf=np.finfo(activations.dtype).max,
            neginf=np.finfo(activations.dtype).min,
        )
    actual_len = activations.shape[0]

    messages = [{"role": m["role"], "content": m["content"]} for m in entry["messages"]]
    segments, total_tokens = build_sentence_segments(tokenizer, messages)

    if not segments:
        return None

    offset = total_tokens - actual_len

    sentence_vecs = []
    for seg in segments:
        act_start = max(0, seg["tok_start"] - offset)
        act_end = min(actual_len, seg["tok_end"] - offset)

        if act_start >= act_end or act_start >= actual_len:
            continue

        vec = activations[act_start:act_end].astype(np.float32).mean(axis=0)
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        sentence_vecs.append(vec)

    if len(sentence_vecs) < 3:
        return None

    sentence_vecs = np.stack(sentence_vecs)
    img = build_heatmap_image(sentence_vecs, num_sentences)
    return img, entry["label"], len(sentence_vecs)


class HeatmapDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.noise_std = 0.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
            img = img.clamp(0.0, 1.0)
        return img, self.labels[idx]


# ── CNN Model ────────────────────────────────────────────────────────

class HeatmapCNN(nn.Module):
    """Lightweight CNN for KxK similarity heatmaps.

    4 conv blocks with BatchNorm + ReLU + MaxPool,
    adaptive pooling to 2x2, 2-layer classifier head.
    """

    def __init__(self, in_channels=3, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Method ───────────────────────────────────────────────────────────

class HeatmapCNNMethod(BaseMethod):
    """
    Heatmap CNN method for binary classification.

    Builds 3-channel KxK similarity heatmaps from sentence-level activation
    means and trains a small CNN. Early stopping uses a held-out validation
    split from the training data.

    Expected data formats:
      train: {"entries": [...], "activations_dir": Path, "tokenizer": tokenizer}
      infer: same format
    """

    def __init__(
        self,
        layer: int = 36,
        num_sentences: int = 30,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 32,
        dropout: float = 0.5,
        label_smoothing: float = 0.1,
        noise_std: float = 0.03,
        patience: int = 15,
        val_fraction: float = 0.15,
        seed: int = 42,
        name: Optional[str] = None,
    ):
        if name is None:
            name = f"heatmap_cnn_L{layer}"
        super().__init__(name)

        self.layer = layer
        self.num_sentences = num_sentences
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.noise_std = noise_std
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed

        self._model: Optional[HeatmapCNN] = None
        self._best_state: Optional[dict] = None

    def _build_images(self, entries, act_dir, tokenizer, desc=""):
        """Build heatmap images from entries + activations."""
        images, labels, info = [], [], []
        skipped = 0
        sent_counts = []

        for i, entry in enumerate(entries):
            result = _build_sample(entry, i, tokenizer, act_dir, self.num_sentences)
            if result is None:
                skipped += 1
                continue

            img, label, n_sents = result
            images.append(img)
            labels.append(label)
            info.append({"index": i, "prompt_name": entry["prompt_name"], "label": label})
            sent_counts.append(n_sents)

            if (i + 1) % 500 == 0:
                print(f"    {desc}: {i+1}/{len(entries)} images built...")

        n_yes = sum(1 for l in labels if l == 1)
        print(f"    {desc}: {len(images)} images ({n_yes} yes_rm, "
              f"{len(images)-n_yes} no_rm), {skipped} skipped")
        if sent_counts:
            print(f"    Sentence counts: mean={np.mean(sent_counts):.1f}, "
                  f"median={np.median(sent_counts):.0f}, "
                  f"min={min(sent_counts)}, max={max(sent_counts)}")

        return images, labels, info

    def train(self, data: Any) -> None:
        """
        Train the heatmap CNN with early stopping on a validation split.

        data: {"entries": list, "activations_dir": Path, "tokenizer": tokenizer}
        """
        import nltk
        for resource in ["punkt_tab", "punkt"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        entries = data["entries"]
        act_dir = Path(data["activations_dir"])
        tokenizer = data["tokenizer"]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"Building heatmap images (K={self.num_sentences})...")
        images, labels, info = self._build_images(entries, act_dir, tokenizer, desc="Train")

        if len(images) < 10:
            raise ValueError(f"Too few samples ({len(images)}), need >= 10")

        # Stratified train/val split
        n_val = max(2, int(len(images) * self.val_fraction))
        indices = np.arange(len(images))
        labels_arr = np.array(labels)

        # Stratify: take val_fraction from each class
        val_idx = []
        for cls in [0, 1]:
            cls_idx = indices[labels_arr == cls]
            np.random.shuffle(cls_idx)
            n_cls_val = max(1, int(len(cls_idx) * self.val_fraction))
            val_idx.extend(cls_idx[:n_cls_val].tolist())
        train_idx = sorted(set(range(len(images))) - set(val_idx))
        val_idx = sorted(val_idx)

        train_images = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        n_train_yes = sum(1 for l in train_labels if l == 1)
        n_val_yes = sum(1 for l in val_labels if l == 1)
        print(f"  Split: {len(train_images)} train ({n_train_yes} yes_rm), "
              f"{len(val_images)} val ({n_val_yes} yes_rm)")

        train_dataset = HeatmapDataset(train_images, train_labels)
        train_dataset.noise_std = self.noise_std
        val_dataset = HeatmapDataset(val_images, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = HeatmapCNN(in_channels=3, dropout=self.dropout).to(device)
        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"  HeatmapCNN: {n_params:,} parameters, device={device}")

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        best_val_auc = -1.0
        self._best_state = None
        best_epoch = -1
        patience_counter = 0
        self.loss_history = []
        self.val_history = []

        print(f"  Training (up to {self.epochs} epochs, patience={self.patience})...")

        for epoch in range(self.epochs):
            # Train
            self._model.train()
            total_loss, n_batches = 0.0, 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(self._model(imgs), lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            # Validate
            val_metrics = self._evaluate_loader(val_loader, device)
            self.val_history.append(val_metrics)

            val_auc = val_metrics.get("auc_roc") or 0.0
            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self._best_state = {k: v.cpu().clone()
                                    for k, v in self._model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or marker:
                print(f"    Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                      f"val_auc={val_auc:.3f} | "
                      f"val_acc={val_metrics['balanced_accuracy']:.3f}{marker}")

            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)
        self._model.eval().cpu()

        print(f"  Best val AUC: {best_val_auc:.3f} at epoch {best_epoch+1}")

        # Save checkpoint
        folder = self.get_folder()
        torch.save(self._best_state or self._model.state_dict(), folder / "model.pt")

        train_info = {
            "n_train": len(train_images),
            "n_val": len(val_images),
            "best_epoch": best_epoch + 1,
            "best_val_auc": best_val_auc,
            "total_epochs": len(self.loss_history),
            "n_params": n_params,
            "loss_history": self.loss_history,
        }
        with open(folder / "train_info.json", "w") as f:
            json.dump(train_info, f, indent=2)

    def _evaluate_loader(self, loader, device):
        """Evaluate model on a DataLoader. Returns metrics dict."""
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score,
            precision_score, recall_score, roc_auc_score,
        )

        self._model.eval()
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                logits = self._model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                all_labels.extend(lbls.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }
        try:
            metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics["auc_roc"] = None

        return metrics

    def infer(self, data: Any) -> List[Dict]:
        """
        Run inference on eval data.

        data: {"entries": list, "activations_dir": Path, "tokenizer": tokenizer}
        """
        import nltk
        for resource in ["punkt_tab", "punkt"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        entries = data["entries"]
        act_dir = Path(data["activations_dir"])
        tokenizer = data["tokenizer"]

        images, labels, info = self._build_images(entries, act_dir, tokenizer, desc="Eval")

        if not images:
            print("WARNING: No valid eval images")
            return []

        eval_dataset = HeatmapDataset(images, labels)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        metrics = self._evaluate_loader(eval_loader, device)
        self._model = self._model.cpu()

        # Get per-sample predictions
        self._model.eval()
        all_probs, all_preds = [], []
        with torch.no_grad():
            for imgs, _ in eval_loader:
                logits = self._model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                all_probs.extend(probs.numpy().tolist())
                all_preds.extend(preds.numpy().tolist())

        results = []
        for i, sample_info in enumerate(info):
            results.append({
                **sample_info,
                "prediction": int(all_preds[i]),
                "prob_yes_rm": float(all_probs[i]),
            })

        # Print and save
        auc_s = f"{metrics['auc_roc']:.3f}" if metrics['auc_roc'] is not None else "N/A"
        print(f"HeatmapCNN eval: bal_acc={metrics['balanced_accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} auc={auc_s}")

        folder = self.get_folder()
        output = {"metrics": metrics, "predictions": results}
        with open(folder / "results.json", "w") as f:
            json.dump(output, f, indent=2)

        return results
