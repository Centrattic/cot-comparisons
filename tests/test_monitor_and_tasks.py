"""Tests for LlmMonitor and task data loading (mock openai)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.conftest import write_scruples_csvs, write_forcing_data


# ---------------------------------------------------------------------------
# LlmMonitor
# ---------------------------------------------------------------------------

class TestLlmMonitor:
    def _make_monitor(self, tmp_dir):
        """Create an LlmMonitor with mocked openai client."""
        from src2.prompts.base import BasePrompt
        from src2.methods import LlmMonitor

        class SimplePrompt(BasePrompt):
            def format(self, row):
                return f"Analyze: {row.get('text', '')}"

            def parse_response(self, response):
                return {"verdict": "yes" if "yes" in response.lower() else "no"}

        prompt = SimplePrompt("simple_test")
        monitor = LlmMonitor(
            prompt=prompt,
            model="test/model",
            max_workers=1,
            api_key="fake-key",
        )

        # Replace real openai client with a full mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes, this is sycophantic."
        mock_client.chat.completions.create.return_value = mock_response
        monitor.client = mock_client

        return monitor

    def test_infer_writes_results(self, tmp_dir, mock_task):
        monitor = self._make_monitor(tmp_dir)
        monitor.set_task(mock_task)
        folder = monitor.get_folder()

        data = [
            {"text": "Story 1", "id": "a1"},
            {"text": "Story 2", "id": "a2"},
            {"text": "Story 3", "id": "a3"},
        ]
        results = monitor.infer(data, verbose=False)

        assert len(results) == 3
        assert all("monitor_prompt" in r for r in results)
        assert all("monitor_response" in r for r in results)
        assert all("monitor_prediction" in r for r in results)

        # Verify files written
        assert (folder / "results.jsonl").exists()
        assert (folder / "results.csv").exists()
        assert (folder / "config.json").exists()

        # Verify JSONL is valid
        with open(folder / "results.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "monitor_prediction" in parsed

        # Verify config
        with open(folder / "config.json") as f:
            config = json.load(f)
        assert config["model"] == "test/model"
        assert config["num_rows"] == 3

    def test_prediction_parsed(self, tmp_dir, mock_task):
        monitor = self._make_monitor(tmp_dir)
        monitor.set_task(mock_task)
        monitor.get_folder()

        results = monitor.infer([{"text": "test"}], verbose=False)
        assert results[0]["monitor_prediction"] == {"verdict": "yes"}


# ---------------------------------------------------------------------------
# ScruplesTask — data loading
# ---------------------------------------------------------------------------

class TestScruplesDataLoading:
    def test_get_data_returns_false_when_empty(self, tmp_dir):
        from src2.tasks import ScruplesTask

        task = ScruplesTask(
            subject_model="test/model",
            data_dir=tmp_dir / "empty",
            api_key="fake",
        )
        assert task.get_data(load=False) is False

    def test_get_data_returns_true_when_exists(self, tmp_dir):
        from src2.tasks import ScruplesTask

        data_dir = tmp_dir / "scruples"
        write_scruples_csvs(data_dir, "first_person")

        task = ScruplesTask(
            subject_model="test/model",
            variant="first_person",
            data_dir=data_dir,
            api_key="fake",
        )
        assert task.get_data(load=False) is True

    def test_get_data_load_returns_dataframes(self, tmp_dir):
        from src2.tasks import ScruplesTask

        data_dir = tmp_dir / "scruples"
        write_scruples_csvs(data_dir, "first_person", n_anecdotes=3, n_runs=2)

        task = ScruplesTask(
            subject_model="test/model",
            variant="first_person",
            data_dir=data_dir,
            api_key="fake",
        )
        data = task.get_data(load=True)
        assert data is not None
        assert "results" in data
        assert "prompts" in data
        assert isinstance(data["results"], pd.DataFrame)
        assert isinstance(data["prompts"], pd.DataFrame)
        assert len(data["prompts"]) == 3
        # 3 anecdotes * 2 arms * 2 runs = 12
        assert len(data["results"]) == 12

    def test_get_activations_returns_paths(self, tmp_dir):
        from src2.tasks import ScruplesTask

        data_dir = tmp_dir / "scruples"
        write_scruples_csvs(data_dir, "first_person", n_anecdotes=2, n_runs=1)

        task = ScruplesTask(
            subject_model="test/model",
            variant="first_person",
            data_dir=data_dir,
            api_key="fake",
        )
        assert task.get_activations(load=False) is True
        paths = task.get_activations(load=True)
        assert paths is not None
        assert len(paths) > 0
        assert all(p.suffix == ".json" for p in paths)


# ---------------------------------------------------------------------------
# ForcingTask — data loading
# ---------------------------------------------------------------------------

class TestForcingDataLoading:
    def test_get_data_returns_false_when_empty(self, tmp_dir):
        from src2.tasks import ForcingTask

        task = ForcingTask(model="test/model", data_dir=tmp_dir / "empty")
        assert task.get_data(load=False) is False

    def test_get_data_returns_true_when_exists(self, tmp_dir):
        from src2.tasks import ForcingTask

        data_dir = tmp_dir / "forcing_data"
        write_forcing_data(data_dir, question_id="test_q1")

        task = ForcingTask(model="test/model", data_dir=data_dir)
        assert task.get_data(load=False) is True

    def test_get_data_load_returns_summaries(self, tmp_dir):
        from src2.tasks import ForcingTask

        data_dir = tmp_dir / "forcing_data"
        write_forcing_data(data_dir, question_id="test_q1", n_sentences=3)

        task = ForcingTask(model="test/model", data_dir=data_dir)
        data = task.get_data(load=True)
        assert data is not None
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "question_id" in data[0]

    def test_get_probe_data_with_activations(self, tmp_dir):
        """Write synthetic .npz alongside force JSONs, verify get_probe_data returns them."""
        from src2.tasks import ForcingTask
        from src2.data_slice import DataSlice

        data_dir = tmp_dir / "forcing_data"
        write_forcing_data(data_dir, question_id="test_q1", n_sentences=2, n_forces=2)

        forcing_dir = data_dir / "forcing"
        # Write companion .npz files for each force_*.json
        for force_json in sorted(forcing_dir.rglob("force_*.json")):
            npz_path = force_json.with_suffix(".npz")
            act = np.random.randn(128).astype(np.float32)
            np.savez(npz_path, layer16_last_thinking=act)

        task = ForcingTask(model="test/model", data_dir=data_dir)
        samples = task.get_probe_data(layer=16, data_slice=DataSlice.all())
        assert len(samples) > 0
        for s in samples:
            assert "activation" in s
            assert "answer_distribution" in s
            assert s["activation"].shape == (128,)
            # Text data linked from companion JSON
            assert "partial_cot" in s
            assert "raw_response" in s
            assert "full_prompt" in s
            assert len(s["full_prompt"]) > 0
            assert "answer" in s


# ---------------------------------------------------------------------------
# Task.evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_scruples_evaluate(self, mock_task):
        predictions = [
            {"answer": "A", "is_sycophantic": True},
            {"answer": "B", "is_sycophantic": False},
            {"answer": "A", "is_sycophantic": True},
        ]
        ground_truth = [True, False, True]
        result = mock_task.evaluate(predictions, ground_truth)
        assert "sycophancy_rate" in result
        assert "accuracy" in result
        assert result["sycophancy_rate"] == pytest.approx(2 / 3)

    def test_scruples_evaluate_empty(self, mock_task):
        result = mock_task.evaluate([], [])
        assert result["sycophancy_rate"] == 0.0
        assert result["accuracy"] == 0.0

    def test_forcing_evaluate(self, tmp_dir):
        from src2.tasks import ForcingTask

        task = ForcingTask(model="test/model", data_dir=tmp_dir / "f")
        result = task.evaluate(["A", "B", "C"], ["A", "B", "D"])
        assert result["accuracy"] == pytest.approx(2 / 3)

    def test_forcing_evaluate_empty(self, tmp_dir):
        from src2.tasks import ForcingTask

        task = ForcingTask(model="test/model", data_dir=tmp_dir / "f")
        result = task.evaluate([], [])
        assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# ForcingTask — full_prompt in force JSONs
# ---------------------------------------------------------------------------

class TestForcingFullPrompt:
    def test_force_json_contains_full_prompt(self, tmp_dir):
        """Verify that synthetic force_*.json files include the full_prompt field."""
        data_dir = tmp_dir / "forcing_data"
        write_forcing_data(data_dir, question_id="test_q1", n_sentences=2, n_forces=2)

        forcing_dir = data_dir / "forcing"
        force_files = sorted(forcing_dir.rglob("force_*.json"))
        assert len(force_files) > 0

        for fp in force_files:
            with open(fp) as f:
                data = json.load(f)
            assert "full_prompt" in data, f"full_prompt missing from {fp}"
            assert len(data["full_prompt"]) > 0, f"full_prompt is empty in {fp}"
            assert "<|im_assistant|>" in data["full_prompt"]
