"""Tests for get_config() and OutputManager."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src2.utils.output import OutputManager


# ---------------------------------------------------------------------------
# Task.get_config()
# ---------------------------------------------------------------------------

class TestTaskGetConfig:
    def test_scruples_config_has_required_keys(self, mock_task):
        config = mock_task.get_config()
        assert "name" in config
        assert "data_dir" in config
        assert isinstance(config["data_dir"], str)
        assert "subject_model" in config
        assert "variant" in config

    def test_scruples_config_values(self, mock_task):
        config = mock_task.get_config()
        assert config["subject_model"] == "openrouter/test/model"
        assert config["variant"] == "first_person"

    def test_config_excludes_private(self, mock_task):
        config = mock_task.get_config()
        for key in config:
            assert not key.startswith("_"), f"Private key {key} leaked into config"

    def test_config_excludes_non_serializable(self, mock_task):
        """client (openai.OpenAI mock) should not appear in config."""
        config = mock_task.get_config()
        assert "client" not in config

    def test_forcing_task_config(self, tmp_dir):
        from src2.tasks import ForcingTask
        task = ForcingTask(model="test/model", data_dir=tmp_dir / "forcing_data")
        config = task.get_config()
        assert config["name"] == "forcing"
        assert "model" in config


# ---------------------------------------------------------------------------
# Method.get_config()
# ---------------------------------------------------------------------------

class TestMethodGetConfig:
    def test_linear_probe_config(self):
        from src2.methods import LinearProbe
        probe = LinearProbe(layer=16, mode="ridge")
        config = probe.get_config()
        assert config["name"].startswith("linear_probe")
        assert "config" in config  # ProbeConfig dataclass
        assert config["config"]["layer"] == 16
        assert config["config"]["mode"] == "ridge"

    def test_attention_probe_config(self):
        from src2.methods import AttentionProbe
        probe = AttentionProbe(layer=32, num_heads=8)
        config = probe.get_config()
        assert config["layer"] == 32
        assert config["num_heads"] == 8

    def test_config_excludes_private_attrs(self):
        from src2.methods import LinearProbe
        probe = LinearProbe(layer=16)
        config = probe.get_config()
        for key in config:
            assert not key.startswith("_"), f"Private key {key} leaked"


# ---------------------------------------------------------------------------
# get_folder() and method_config.json
# ---------------------------------------------------------------------------

class TestGetFolder:
    def test_creates_method_config_json(self, mock_task):
        from src2.methods import LinearProbe
        probe = LinearProbe(layer=16, mode="ridge")
        probe.set_task(mock_task)
        folder = probe.get_folder()

        config_path = folder / "method_config.json"
        assert config_path.exists()

    def test_method_config_has_method_and_task(self, mock_task):
        from src2.methods import LinearProbe
        probe = LinearProbe(layer=16, mode="ridge")
        probe.set_task(mock_task)
        folder = probe.get_folder()

        with open(folder / "method_config.json") as f:
            config = json.load(f)

        assert "method" in config
        assert "task" in config
        assert config["method"]["name"].startswith("linear_probe")
        assert config["task"]["variant"] == "first_person"

    def test_method_config_is_valid_json(self, mock_task):
        from src2.methods import LinearProbe
        probe = LinearProbe(layer=16, mode="ridge")
        probe.set_task(mock_task)
        folder = probe.get_folder()

        with open(folder / "method_config.json") as f:
            config = json.load(f)  # Will raise if invalid JSON

        assert isinstance(config, dict)


# ---------------------------------------------------------------------------
# OutputManager
# ---------------------------------------------------------------------------

class TestOutputManager:
    def test_create_run_folder(self, tmp_dir):
        om = OutputManager(tmp_dir / "output")
        folder = om.create_run_folder()
        assert folder.exists()
        assert folder.is_dir()
        assert folder.parent == tmp_dir / "output"

    def test_mark_success_creates_symlink(self, tmp_dir):
        om = OutputManager(tmp_dir / "output")
        folder = om.create_run_folder()
        om.mark_success()

        latest = tmp_dir / "output" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == folder.resolve()

    def test_mark_success_updates_symlink(self, tmp_dir):
        import time
        om = OutputManager(tmp_dir / "output")

        folder1 = om.create_run_folder()
        om.mark_success()

        time.sleep(1.1)  # Ensure different timestamp

        folder2 = om.create_run_folder()
        om.mark_success()

        latest = tmp_dir / "output" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == folder2.resolve()
        assert folder2 != folder1
