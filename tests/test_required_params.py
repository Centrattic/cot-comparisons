"""Tests that verify TypeError when required params are missing."""

import pytest


class TestScruplesTaskRequired:
    def test_missing_subject_model(self):
        from src2.tasks import ScruplesTask
        with pytest.raises(TypeError, match="subject_model"):
            ScruplesTask()


class TestForcingTaskRequired:
    def test_missing_model(self):
        from src2.tasks import ForcingTask
        with pytest.raises(TypeError, match="model"):
            ForcingTask()


class TestResamplingTaskRequired:
    def test_missing_model(self):
        from src2.tasks import ResamplingTask
        with pytest.raises(TypeError, match="model"):
            ResamplingTask()


class TestLinearProbeRequired:
    def test_missing_layer(self):
        from src2.methods import LinearProbe
        with pytest.raises(TypeError, match="layer"):
            LinearProbe()


class TestAttentionProbeRequired:
    def test_missing_layer(self):
        from src2.methods import AttentionProbe
        with pytest.raises(TypeError, match="layer"):
            AttentionProbe()


class TestContrastiveSAERequired:
    def test_missing_sae_repo(self):
        from src2.methods import ContrastiveSAE
        with pytest.raises(TypeError):
            ContrastiveSAE(sae_layer=16)

    def test_missing_sae_layer(self):
        from src2.methods import ContrastiveSAE
        with pytest.raises(TypeError):
            ContrastiveSAE(sae_repo="some/repo")


class TestLlmMonitorRequired:
    def test_missing_model(self, tmp_dir):
        """LlmMonitor requires both prompt and model."""
        from src2.prompts.base import BasePrompt

        class DummyPrompt(BasePrompt):
            def format(self, row):
                return str(row)
            def parse_response(self, response):
                return response

        prompt = DummyPrompt("test")
        from src2.methods import LlmMonitor
        with pytest.raises(TypeError):
            LlmMonitor(prompt=prompt)


class TestExtractActivationsRequired:
    def test_missing_model_name(self, mock_task):
        """extract_activations requires model_name and layers."""
        with pytest.raises(TypeError):
            mock_task.extract_activations()


class TestGetProbeDataRequired:
    def test_missing_layer(self, mock_task):
        """get_probe_data requires layer parameter."""
        with pytest.raises(TypeError):
            mock_task.get_probe_data()
