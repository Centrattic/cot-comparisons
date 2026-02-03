"""
Attention-weighted pooling probe (Eleuther AI style).

Takes activations over ALL sequence positions (not just one position).
Uses learned attention weights to pool over the sequence before classification.

Architecture:
  - query_proj: hidden_dim -> num_heads (learned attention logits per head)
  - Position bias: learned ALiBi-style relative position weighting
  - value_proj: hidden_dim -> output_dim (per head)
  - Softmax attention -> weighted sum of values -> classification

This is essentially cross-attention with one learned query token.
Advantage: can learn which positions in the CoT are informative
rather than using a fixed position.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseMethod


class AttentionPoolingProbe(nn.Module):
    """
    Attention-weighted pooling over sequence positions followed by classification.

    For each head:
      1. Compute attention logits: a_t = query_proj(h_t) + position_bias(t)
      2. Softmax over positions: w_t = softmax(a)
      3. Weighted sum of value projections: out = sum(w_t * value_proj(h_t))
    Concatenate heads and project to output.
    """

    def __init__(
        self,
        hidden_dim: int = 5120,
        num_heads: int = 4,
        output_dim: int = 1,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, num_heads)

        self.position_bias = nn.Parameter(torch.zeros(num_heads, max_seq_len))

        self.value_proj = nn.Linear(hidden_dim, self.head_dim * num_heads)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.head_dim * num_heads, output_dim),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] boolean mask (True = valid)

        Returns:
            [batch, output_dim]
        """
        batch, seq_len, _ = x.shape

        # size: [batch, seq_len, num_heads]
        attn_logits = self.query_proj(x)

        # size: [1, seq_len, num_heads]
        pos_bias = self.position_bias[:, :seq_len].T.unsqueeze(0)

        attn_logits = attn_logits + pos_bias

        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # softmax over seq_len
        attn_weights = torch.softmax(attn_logits, dim=1)

        # size: [batch, seq_len, head_dim * num_heads]
        values = self.value_proj(x)

        # reshape: [batch, seq_len, num_heads, head_dim]
        values = values.view(batch, seq_len, self.num_heads, self.head_dim)

        # weighted sum size: [batch, num_heads, head_dim]
        pooled = (attn_weights.unsqueeze(-1) * values).sum(dim=1)

        # concatenate heads: [batch, head_dim * num_heads]
        pooled = pooled.view(batch, -1)

        # [batch, output_dim]
        return self.classifier(pooled)


class AttentionProbe(BaseMethod):
    """
    Attention-weighted pooling probe method.

    Receives pre-extracted full-sequence activations and uses
    learned attention to identify which positions are informative.

    Supports two modes:
      - "regression": MSELoss, continuous y values (default, backward compat)
      - "classification": CrossEntropyLoss, integer class labels

    Expected data formats:
      train: {"X_list": List[ndarray], "y": ndarray}
      infer: {"X_list": List[ndarray], "anecdote_ids": list}
    """

    def __init__(
        self,
        layer: int,
        mode: str = "regression",
        num_heads: int = 4,
        lr: float = 1e-3,
        epochs: int = 100,
        name: Optional[str] = None,
    ):
        if mode not in ("regression", "classification"):
            raise ValueError(f"mode must be 'regression' or 'classification', got '{mode}'")
        if name is None:
            name = f"attention_probe_L{layer}"
        super().__init__(name)

        self.layer = layer
        self.mode = mode
        self.num_heads = num_heads
        self.lr = lr
        self.epochs = epochs
        self._probe: Optional[AttentionPoolingProbe] = None
        self._num_classes: Optional[int] = None

    def train(self, data: Any) -> None:
        """
        Train the attention probe.

        data: {"X_list": List[ndarray [seq_len, hidden_dim]], "y": ndarray [n]}

        In regression mode, y should be float values.
        In classification mode, y should be integer class labels.
        """
        X_list = data["X_list"]
        y_list = data["y"]

        if len(X_list) < 5:
            raise ValueError(f"Need >= 5 samples, got {len(X_list)}")

        # Pad sequences and create tensors
        max_len = max(x.shape[0] for x in X_list)
        hidden_dim = X_list[0].shape[1]
        batch_size = len(X_list)

        X_padded = torch.zeros(batch_size, max_len, hidden_dim)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, x in enumerate(X_list):
            sl = x.shape[0]
            X_padded[i, :sl, :] = torch.from_numpy(x)
            mask[i, :sl] = True

        if self.mode == "classification":
            y = torch.tensor(y_list, dtype=torch.long)
            num_classes = int(y.max().item()) + 1
            self._num_classes = num_classes
            output_dim = num_classes
            loss_fn = nn.CrossEntropyLoss()
        else:
            y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(-1)
            output_dim = 1
            loss_fn = nn.MSELoss()

        # Train probe
        self._probe = AttentionPoolingProbe(
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            output_dim=output_dim,
            max_seq_len=max_len,
        )

        optimizer = torch.optim.Adam(self._probe.parameters(), lr=self.lr)

        self._probe.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self._probe(X_padded, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        self._probe.eval()

        # Save checkpoint
        folder = self._output.run_folder
        torch.save(self._probe.state_dict(), folder / "probe.pt")
        config = {
            "layer": self.layer,
            "mode": self.mode,
            "num_heads": self.num_heads,
            "hidden_dim": hidden_dim,
            "max_seq_len": max_len,
            "n_samples": batch_size,
            "final_loss": float(loss.item()),
        }
        if self.mode == "classification":
            config["num_classes"] = self._num_classes
        with open(folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(
            f"AttentionProbe trained: {batch_size} samples, "
            f"mode={self.mode}, final loss={loss.item():.4f}"
        )

    def infer(self, data: Any) -> Any:
        """
        Run inference with trained attention probe.

        data: {"X_list": List[ndarray], "anecdote_ids": list}

        In regression mode, returns predicted_switch_rate per sample.
        In classification mode, returns predicted_class and class_probabilities.
        """
        if self._probe is None:
            raise RuntimeError("Probe not trained. Call train() first.")

        X_list = data["X_list"]
        anecdote_ids = data.get("anecdote_ids", list(range(len(X_list))))

        results = []
        self._probe.eval()

        for i, act in enumerate(X_list):
            seq_len, hidden_dim = act.shape
            x = torch.from_numpy(act).float().unsqueeze(0)
            mask = torch.ones(1, seq_len, dtype=torch.bool)

            with torch.no_grad():
                pred = self._probe(x, mask)

                if self.mode == "classification":
                    probs = torch.softmax(pred, dim=-1).squeeze(0)
                    predicted_class = int(probs.argmax().item())
                    results.append(
                        {
                            "anecdote_id": anecdote_ids[i],
                            "predicted_class": predicted_class,
                            "class_probabilities": probs.tolist(),
                        }
                    )
                else:
                    predicted_switch_rate = float(pred.squeeze())
                    results.append(
                        {
                            "anecdote_id": anecdote_ids[i],
                            "predicted_switch_rate": predicted_switch_rate,
                        }
                    )

        # Save results
        if results and self._output and self._output.run_folder:
            with open(self._output.run_folder / "predictions.json", "w") as f:
                json.dump(results, f, indent=2)

        return results
