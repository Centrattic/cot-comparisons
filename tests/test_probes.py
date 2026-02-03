"""End-to-end tests for probes with synthetic data (no mocking needed — pure numpy/torch)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_probe_with_task(probe_cls, tmp_dir, **kwargs):
    """Create a probe and bind it to a minimal mock task so get_folder() works."""
    probe = probe_cls(**kwargs)

    # Minimal task-like object
    task = MagicMock()
    task.data_dir = tmp_dir / "task_data"
    task.data_dir.mkdir(parents=True, exist_ok=True)
    task.get_config.return_value = {"name": "mock_task"}

    probe.set_task(task)
    probe.get_folder()
    return probe


# ---------------------------------------------------------------------------
# LinearProbe — ridge mode
# ---------------------------------------------------------------------------

class TestLinearProbeRidge:
    def test_train_and_infer(self, tmp_dir, sample_activations):
        from src2.methods import LinearProbe

        probe = _setup_probe_with_task(
            LinearProbe, tmp_dir, layer=16, mode="ridge"
        )

        rng = np.random.RandomState(42)
        y = rng.rand(10).astype(np.float32)
        probe.train({"X": sample_activations, "y": y})

        # Verify probe.json saved
        folder = probe._output.run_folder
        assert (folder / "probe.json").exists()
        with open(folder / "probe.json") as f:
            probe_data = json.load(f)
        assert "weights" in probe_data
        assert "cv_metrics" in probe_data

        # Infer
        X_test = rng.randn(5, 128).astype(np.float32)
        ids = [f"id_{i}" for i in range(5)]
        results = probe.infer({"X": X_test, "anecdote_ids": ids})
        assert len(results) == 5
        assert all("predicted_switch_rate" in r for r in results)
        assert all("anecdote_id" in r for r in results)

    def test_too_few_samples_raises(self, tmp_dir):
        from src2.methods import LinearProbe

        probe = _setup_probe_with_task(
            LinearProbe, tmp_dir, layer=16, mode="ridge"
        )

        X = np.random.randn(2, 128).astype(np.float32)
        y = np.array([0.1, 0.9], dtype=np.float32)
        with pytest.raises(ValueError, match="Need >= 3"):
            probe.train({"X": X, "y": y})


# ---------------------------------------------------------------------------
# LinearProbe — soft_ce mode
# ---------------------------------------------------------------------------

class TestLinearProbeSoftCE:
    def test_train_and_infer(self, tmp_dir):
        from src2.methods import LinearProbe

        probe = _setup_probe_with_task(
            LinearProbe, tmp_dir, layer=16, mode="soft_ce"
        )

        rng = np.random.RandomState(42)
        data = []
        for _ in range(20):
            dist_raw = rng.dirichlet([1, 1, 1, 1])
            data.append({
                "activation": rng.randn(128).astype(np.float32),
                "answer_distribution": {
                    "A": float(dist_raw[0]),
                    "B": float(dist_raw[1]),
                    "C": float(dist_raw[2]),
                    "D": float(dist_raw[3]),
                },
            })

        probe.train(data)

        folder = probe._output.run_folder
        assert (folder / "probe.pt").exists()
        assert (folder / "scaler.npz").exists()
        assert (folder / "config.json").exists()

        # Infer
        test_data = [
            {"activation": rng.randn(128).astype(np.float32)}
            for _ in range(5)
        ]
        results = probe.infer(test_data)
        assert len(results) == 5
        for r in results:
            assert "predicted_distribution" in r
            dist = r["predicted_distribution"]
            assert set(dist.keys()) == {"A", "B", "C", "D"}
            assert abs(sum(dist.values()) - 1.0) < 1e-5

    def test_infer_before_train_raises(self, tmp_dir):
        from src2.methods import LinearProbe

        probe = _setup_probe_with_task(
            LinearProbe, tmp_dir, layer=16, mode="soft_ce"
        )
        with pytest.raises(RuntimeError, match="not trained"):
            probe.infer([{"activation": np.zeros(128)}])


# ---------------------------------------------------------------------------
# AttentionProbe
# ---------------------------------------------------------------------------

class TestAttentionProbe:
    def test_train_and_infer(self, tmp_dir):
        from src2.methods import AttentionProbe

        probe = _setup_probe_with_task(
            AttentionProbe, tmp_dir, layer=16, num_heads=2, epochs=10
        )

        rng = np.random.RandomState(42)
        X_list = [rng.randn(rng.randint(20, 60), 128).astype(np.float32) for _ in range(10)]
        y = rng.rand(10).astype(np.float32)

        probe.train({"X_list": X_list, "y": y})

        folder = probe._output.run_folder
        assert (folder / "probe.pt").exists()
        assert (folder / "config.json").exists()

        # Infer
        X_test = [rng.randn(30, 128).astype(np.float32) for _ in range(5)]
        ids = [f"id_{i}" for i in range(5)]
        results = probe.infer({"X_list": X_test, "anecdote_ids": ids})
        assert len(results) == 5
        assert all("predicted_switch_rate" in r for r in results)

    def test_too_few_samples_raises(self, tmp_dir):
        from src2.methods import AttentionProbe

        probe = _setup_probe_with_task(
            AttentionProbe, tmp_dir, layer=16, num_heads=2, epochs=10
        )

        X_list = [np.random.randn(20, 128).astype(np.float32) for _ in range(3)]
        y = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        with pytest.raises(ValueError, match="Need >= 5"):
            probe.train({"X_list": X_list, "y": y})

    def test_infer_before_train_raises(self, tmp_dir):
        from src2.methods import AttentionProbe

        probe = _setup_probe_with_task(
            AttentionProbe, tmp_dir, layer=16, num_heads=2
        )
        with pytest.raises(RuntimeError, match="not trained"):
            probe.infer({"X_list": [np.zeros((10, 128))], "anecdote_ids": ["a"]})


# ---------------------------------------------------------------------------
# ContrastiveSAE (mock the SAE model loading)
# ---------------------------------------------------------------------------

class TestContrastiveSAE:
    def _make_mock_sae(self, activation_dim=128, dict_size=256):
        """Build a small synthetic BatchTopKSAE on CPU."""
        from src2.methods.contrastive_sae import BatchTopKSAE

        sae = BatchTopKSAE(activation_dim=activation_dim, dict_size=dict_size, k=32)
        sae.eval()
        return sae

    def test_train_and_infer(self, tmp_dir, sample_activations):
        from src2.methods import ContrastiveSAE

        probe = _setup_probe_with_task(
            ContrastiveSAE, tmp_dir,
            sae_repo="test/repo", sae_layer=16, top_k_features=10,
        )

        mock_sae = self._make_mock_sae()
        with patch.object(
            type(probe), '_get_sae', return_value=mock_sae
        ):
            rng = np.random.RandomState(42)
            y = rng.rand(10).astype(np.float32)
            probe.train({"X": sample_activations, "y": y})

            folder = probe._output.run_folder
            assert (folder / "features.json").exists()
            assert (folder / "probe.json").exists()

            with open(folder / "features.json") as f:
                features = json.load(f)
            assert len(features) <= 10
            assert all("feature_idx" in feat for feat in features)

            # Infer
            X_test = rng.randn(5, 128).astype(np.float32)
            results = probe.infer({
                "X": X_test,
                "anecdote_ids": [f"id_{i}" for i in range(5)],
            })
            assert len(results) == 5
            assert all("predicted_switch_rate" in r for r in results)

    def test_infer_before_train_raises(self, tmp_dir):
        from src2.methods import ContrastiveSAE

        probe = _setup_probe_with_task(
            ContrastiveSAE, tmp_dir,
            sae_repo="test/repo", sae_layer=16,
        )
        with pytest.raises(RuntimeError, match="Not trained"):
            probe.infer({"X": np.zeros((5, 128)), "anecdote_ids": ["a"] * 5})
