"""
Contrastive SAE analysis method.

Uses Sparse Autoencoders (SAEs) trained on Qwen3-32B to find features that
correlate with sycophantic behavior. Trains a ridge regression probe on
SAE feature deltas (intervention - control) to predict switch_rate.

SAE source: https://huggingface.co/adamkarvonen/qwen3-32b-saes
Architecture: BatchTopK SAEs (JumpReLU inference mode)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseMethod

HIGH_NORM_MULTIPLIER = 10.0


class BatchTopKSAE(nn.Module):
    """
    BatchTopK SAE matching the dictionary_learning implementation.
    Uses learned threshold (JumpReLU) at inference.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor, use_threshold: bool = True) -> torch.Tensor:
        post_relu = nn.functional.relu(self.encoder(x - self.b_dec))
        if use_threshold:
            return post_relu * (post_relu > self.threshold)
        flattened = post_relu.flatten()
        topk = flattened.topk(self.k.item() * x.size(0), sorted=False, dim=-1)
        encoded = torch.zeros_like(flattened).scatter_(-1, topk.indices, topk.values)
        return encoded.reshape(post_relu.shape)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded = self.encode(x)
        x_hat = self.decode(encoded)
        if output_features:
            return x_hat, encoded
        return x_hat

    @classmethod
    def from_pretrained(cls, repo_id: str, layer: int,
                        trainer: int = 0, device="cuda") -> "BatchTopKSAE":
        from huggingface_hub import hf_hub_download

        subdir = f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}"
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/config.json")
        with open(config_path) as f:
            config = json.load(f)

        tc = config["trainer"]
        sae = cls(activation_dim=tc["activation_dim"], dict_size=tc["dict_size"], k=tc["k"])
        weights_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/ae.pt")
        sae.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        sae = sae.to(device)
        sae.eval()
        print(f"SAE loaded: layer={layer}, trainer={trainer}, "
              f"dict_size={tc['dict_size']}, k={tc['k']}")
        return sae


class ContrastiveSAE(BaseMethod):
    """
    SAE-based contrastive analysis method.

    Takes pre-computed contrastive activation deltas (same format as LinearProbe),
    encodes them through the SAE, finds correlated features, and trains ridge regression.

    Expected data formats:
      train: {"X": ndarray [n, hidden_dim], "y": ndarray [n]}
      infer: {"X": ndarray [n, hidden_dim], "anecdote_ids": list}
    """

    def __init__(
        self,
        sae_repo: str,
        sae_layer: int,
        sae_trainer: int = 0,
        top_k_features: int = 50,
        ridge_alpha: float = 1.0,
        name: Optional[str] = None,
    ):
        if name is None:
            name = f"contrastive_sae_L{sae_layer}_T{sae_trainer}"
        super().__init__(name)

        self.sae_repo = sae_repo
        self.sae_layer = sae_layer
        self.sae_trainer = sae_trainer
        self.top_k_features = top_k_features
        self.ridge_alpha = ridge_alpha

        self._sae: Optional[BatchTopKSAE] = None

        # Trained state
        self._feature_indices: Optional[np.ndarray] = None
        self._weights = None
        self._intercept = None
        self._scaler_mean = None
        self._scaler_scale = None

    def _get_sae(self) -> BatchTopKSAE:
        if self._sae is None:
            self._sae = BatchTopKSAE.from_pretrained(
                repo_id=self.sae_repo,
                layer=self.sae_layer, trainer=self.sae_trainer,
            )
        return self._sae

    def _encode_through_sae(self, X: np.ndarray) -> np.ndarray:
        """Encode activation deltas through SAE. Returns [n, dict_size]."""
        sae = self._get_sae()
        device = next(sae.parameters()).device
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            features = sae.encode(X_tensor)
        return features.cpu().numpy()

    def train(self, data: Any) -> None:
        """
        Find correlated SAE features and train ridge regression.

        data: {"X": ndarray [n, hidden_dim], "y": ndarray [n]}
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict, LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import pearsonr

        X_raw = data["X"]
        y = data["y"]

        if len(X_raw) < 3:
            raise ValueError(f"Need >= 3 samples, got {len(X_raw)}")

        # Encode through SAE
        X_full = self._encode_through_sae(X_raw)

        # Find top-K correlated features
        correlations = []
        for fi in range(X_full.shape[1]):
            if np.std(X_full[:, fi]) > 0:
                r, p = pearsonr(X_full[:, fi], y)
                correlations.append((fi, abs(r), r, p))

        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = correlations[:self.top_k_features]
        self._feature_indices = np.array([f[0] for f in top_features])

        # Save features info
        folder = self._output.run_folder
        features_info = [
            {"feature_idx": int(fi), "abs_pearson_r": float(ar),
             "pearson_r": float(r), "p_value": float(p)}
            for fi, ar, r, p in top_features
        ]
        with open(folder / "features.json", "w") as f:
            json.dump(features_info, f, indent=2)

        # Train ridge on selected features
        X_selected = X_full[:, self._feature_indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        model = Ridge(alpha=self.ridge_alpha)
        y_pred_cv = cross_val_predict(model, X_scaled, y, cv=LeaveOneOut())
        model.fit(X_scaled, y)

        self._weights = model.coef_
        self._intercept = float(model.intercept_)
        self._scaler_mean = scaler.mean_
        self._scaler_scale = scaler.scale_

        ss_res = np.sum((y - y_pred_cv) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r, _ = pearsonr(y_pred_cv, y)

        probe_data = {
            "feature_indices": self._feature_indices.tolist(),
            "weights": self._weights.tolist(),
            "intercept": self._intercept,
            "scaler_mean": self._scaler_mean.tolist(),
            "scaler_scale": self._scaler_scale.tolist(),
            "cv_metrics": {"r_squared": float(r_squared), "pearson_r": float(r),
                           "mse": float(np.mean((y_pred_cv - y) ** 2))},
        }
        with open(folder / "probe.json", "w") as f:
            json.dump(probe_data, f, indent=2)

        print(f"ContrastiveSAE trained: {len(top_features)} features, R²={r_squared:.3f}, r={r:.3f}")

    def infer(self, data: Any) -> Any:
        """
        Run inference with trained SAE probe.

        data: {"X": ndarray [n, hidden_dim], "anecdote_ids": list}
        Returns list of dicts with predicted_switch_rate per anecdote.
        """
        if self._weights is None:
            raise RuntimeError("Not trained. Call train() first.")

        X_raw = data["X"]
        anecdote_ids = data.get("anecdote_ids", list(range(len(X_raw))))

        # Encode through SAE
        X_full = self._encode_through_sae(X_raw)

        results = []
        for i in range(len(X_raw)):
            selected = X_full[i, self._feature_indices]
            x_scaled = (selected - self._scaler_mean) / self._scaler_scale
            pred = float(np.dot(x_scaled, self._weights) + self._intercept)
            results.append({
                "anecdote_id": anecdote_ids[i],
                "predicted_switch_rate": pred,
            })

        # Save
        if results and self._output and self._output.run_folder:
            with open(self._output.run_folder / "predictions.json", "w") as f:
                json.dump(results, f, indent=2)

        return results
