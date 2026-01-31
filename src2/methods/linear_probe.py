"""
Single-position linear probe method.

Supports two modes:
- ridge: Ridge regression on contrastive activation deltas to predict switch_rate
         (scruples task). Uses leave-one-out CV.
- soft_ce: Linear/MLP classifier on single-position activations predicting
           answer distributions via soft cross-entropy (forced response task).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseMethod

@dataclass
class ProbeConfig:
    """Configuration for a single probe."""
    layer: int
    token_position: str = "last_thinking"
    mode: str = "ridge"  # "ridge" or "soft_ce"

    def __str__(self) -> str:
        return f"layer{self.layer}_{self.token_position}_{self.mode}"


class LinearProbe(BaseMethod):
    """
    Single-position activation probe.

    Modes:
      - ridge: contrastive activation deltas -> switch_rate (scruples)
      - soft_ce: single-position activations -> answer distribution (forced response)

    Expected data formats:
      ridge train:   {"X": ndarray [n, hidden_dim], "y": ndarray [n]}
      ridge infer:   {"X": ndarray [n, hidden_dim], "anecdote_ids": list}
      soft_ce train: list of dicts with "activation" and "answer_distribution"
      soft_ce infer: list of dicts with "activation"
    """

    def __init__(
        self,
        layer: int,
        mode: str = "ridge",
        token_position: str = "last_thinking",
        ridge_alpha: float = 1.0,
        name: Optional[str] = None,
    ):
        if name is None:
            name = f"linear_probe_{mode}_L{layer}_{token_position}"
        super().__init__(name)

        self.config = ProbeConfig(
            layer=layer, token_position=token_position, mode=mode,
        )
        self.ridge_alpha = ridge_alpha

        # Trained probe state
        self._weights = None
        self._intercept = None
        self._scaler_mean = None
        self._scaler_scale = None

    def train(self, data: Any) -> None:
        """
        Train the probe.

        For ridge mode: data = {"X": ndarray, "y": ndarray}
        For soft_ce mode: data = list of dicts with "activation" and "answer_distribution"
        """
        if self.config.mode == "ridge":
            self._train_ridge(data)
        else:
            self._train_soft_ce(data)

    def _train_ridge(self, data: Any) -> None:
        """Train ridge regression on contrastive activation deltas."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict, LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import pearsonr

        X = data["X"]
        y = data["y"]

        if len(X) < 3:
            raise ValueError(f"Need >= 3 samples for ridge training, got {len(X)}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=self.ridge_alpha)
        y_pred_cv = cross_val_predict(model, X_scaled, y, cv=LeaveOneOut())
        model.fit(X_scaled, y)

        # Save state
        self._weights = model.coef_
        self._intercept = float(model.intercept_)
        self._scaler_mean = scaler.mean_
        self._scaler_scale = scaler.scale_

        # Compute metrics
        ss_res = np.sum((y - y_pred_cv) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r, _ = pearsonr(y_pred_cv, y)

        # Save to folder
        folder = self._output.run_folder
        probe_data = {
            "config": {"layer": self.config.layer, "token_position": self.config.token_position},
            "weights": self._weights.tolist(),
            "intercept": self._intercept,
            "scaler_mean": self._scaler_mean.tolist(),
            "scaler_scale": self._scaler_scale.tolist(),
            "cv_metrics": {"r_squared": float(r_squared), "pearson_r": float(r),
                           "mse": float(np.mean((y_pred_cv - y) ** 2)),
                           "mae": float(np.mean(np.abs(y_pred_cv - y)))},
        }
        with open(folder / "probe.json", "w") as f:
            json.dump(probe_data, f, indent=2)

        print(f"Ridge probe trained: R²={r_squared:.3f}, r={r:.3f}")

    def _train_soft_ce(self, data: Any) -> None:
        """
        Train soft cross-entropy classifier for answer distribution prediction.

        data: list of dicts, each with keys:
          - "activation": np.ndarray [hidden_dim]
          - "answer_distribution": dict like {"A": 0.3, "B": 0.5, ...}
          - "question_type": "multiple_choice" or "binary_judge" (optional)
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        if not data:
            raise ValueError("No samples provided for training")

        # Determine labels from first sample
        question_type = data[0].get("question_type", "multiple_choice")
        if question_type == "binary_judge":
            labels = ["YES", "NO"]
        else:
            labels = ["A", "B", "C", "D"]

        n_classes = len(labels)

        # Prepare data
        X = np.stack([s["activation"] for s in data])
        y = np.array([
            [s["answer_distribution"].get(label, 0.0) for label in labels]
            for s in data
        ])

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        self._scaler_mean = scaler.mean_
        self._scaler_scale = scaler.scale_
        self._soft_ce_labels = labels

        device = "cuda" if torch.cuda.is_available() else "cpu"

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

        hidden_dim = X_train.shape[1]

        # Simple linear probe
        probe = torch.nn.Linear(hidden_dim, n_classes).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

        def soft_cross_entropy(logits, soft_labels):
            log_probs = torch.log_softmax(logits, dim=-1)
            return -(soft_labels * log_probs).sum(dim=-1).mean()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(100):
            probe.train()
            optimizer.zero_grad()
            loss = soft_cross_entropy(probe(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_loss = soft_cross_entropy(probe(X_val_t), y_val_t)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()

        self._soft_ce_probe = probe.cpu()
        self._soft_ce_device = device

        # Evaluate
        with torch.no_grad():
            val_logits = probe.to(device)(X_val_t)
            val_probs = torch.softmax(val_logits, dim=-1)
            pred_labels = val_probs.argmax(dim=-1)
            true_labels = y_val_t.argmax(dim=-1)
            accuracy = (pred_labels == true_labels).float().mean().item()

            eps = 1e-8
            kl_div = (
                (y_val_t * (torch.log(y_val_t + eps) - torch.log(val_probs + eps)))
                .sum(dim=-1).mean().item()
            )

        # Save
        folder = self._output.run_folder
        torch.save(best_state or probe.state_dict(), folder / "probe.pt")
        np.savez(folder / "scaler.npz", mean=self._scaler_mean, scale=self._scaler_scale)

        config = {
            "mode": "soft_ce",
            "layer": self.config.layer,
            "token_position": self.config.token_position,
            "question_type": question_type,
            "labels": labels,
            "hidden_dim": hidden_dim,
            "n_classes": n_classes,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "val_accuracy": accuracy,
            "val_kl_divergence": kl_div,
        }
        with open(folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Soft CE probe trained: accuracy={accuracy:.3f}, KL={kl_div:.4f}")

    def infer(self, data: Any) -> Any:
        """
        Run inference.

        For ridge mode: data = {"X": ndarray, "anecdote_ids": list}
        For soft_ce mode: data = list of dicts with "activation"
        """
        if self.config.mode == "ridge":
            return self._infer_ridge(data)
        return self._infer_soft_ce(data)

    def _infer_ridge(self, data: Any) -> List[Dict]:
        """Predict switch_rate from activation deltas."""
        if self._weights is None:
            raise RuntimeError("Probe not trained. Call train() first.")

        X = data["X"]
        anecdote_ids = data.get("anecdote_ids", list(range(len(X))))

        results = []
        for i in range(len(X)):
            x_scaled = (X[i] - self._scaler_mean) / self._scaler_scale
            pred = float(np.dot(x_scaled, self._weights) + self._intercept)
            results.append({
                "anecdote_id": anecdote_ids[i],
                "predicted_switch_rate": pred,
            })

        # Save results
        if results and self._output and self._output.run_folder:
            with open(self._output.run_folder / "predictions.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

        return results

    def _infer_soft_ce(self, data: Any) -> List[Dict]:
        """Predict answer distributions from activations."""
        if not hasattr(self, "_soft_ce_probe") or self._soft_ce_probe is None:
            raise RuntimeError("Probe not trained. Call train() first.")

        labels = self._soft_ce_labels
        results = []

        if isinstance(data, list) and data and "activation" in data[0]:
            device = self._soft_ce_device
            probe = self._soft_ce_probe.to(device)
            probe.eval()

            for row in data:
                act = np.array(row["activation"])
                x_scaled = (act - self._scaler_mean) / self._scaler_scale
                x_t = torch.tensor(x_scaled, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = probe(x_t)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                pred_dist = {label: float(probs[i]) for i, label in enumerate(labels)}
                results.append({**row, "predicted_distribution": pred_dist})
        else:
            print("Warning: soft_ce infer() expects 'activation' key in each row")

        # Save results
        if results and self._output and self._output.run_folder:
            save_results = []
            for r in results:
                save_r = {k: v for k, v in r.items() if k != "activation"}
                save_results.append(save_r)
            with open(self._output.run_folder / "predictions.json", "w") as f:
                json.dump(save_results, f, indent=2, default=str)

        return results
