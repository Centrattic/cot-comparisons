"""Tests for CompressedCotTask and compressed CoT prompts."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.conftest import write_compressed_cot_data


# ---------------------------------------------------------------------------
# CompressedCotTask — construction
# ---------------------------------------------------------------------------

class TestCompressedCotTaskRequired:
    def test_missing_model(self):
        from src2.tasks import CompressedCotTask
        with pytest.raises(TypeError, match="model"):
            CompressedCotTask()

    def test_default_compression_factor(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")
        assert task.compression_factor == 10
        assert task.char_limit_multiplier == 1.5

    def test_custom_compression_params(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(
            model="test/model",
            compression_factor=5,
            char_limit_multiplier=2.0,
            data_dir=tmp_dir / "cot",
        )
        assert task.compression_factor == 5
        assert task.char_limit_multiplier == 2.0


# ---------------------------------------------------------------------------
# CompressedCotTask — get_config()
# ---------------------------------------------------------------------------

class TestCompressedCotConfig:
    def test_config_has_required_keys(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")
        config = task.get_config()
        assert config["name"] == "compressed_cot"
        assert "data_dir" in config
        assert isinstance(config["data_dir"], str)
        assert config["model"] == "test/model"
        assert config["compression_factor"] == 10

    def test_config_excludes_private(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")
        config = task.get_config()
        for key in config:
            assert not key.startswith("_"), f"Private key {key} leaked into config"


# ---------------------------------------------------------------------------
# CompressedCotTask — data loading
# ---------------------------------------------------------------------------

class TestCompressedCotDataLoading:
    def test_get_data_returns_false_when_empty(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "empty")
        assert task.get_data(load=False) is False

    def test_get_data_returns_true_when_exists(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1")
        task = CompressedCotTask(model="test/model", data_dir=data_dir)
        assert task.get_data(load=False) is True

    def test_get_data_load_returns_list(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)
        data = task.get_data(load=True)
        assert data is not None
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "question_id" in data[0]
        assert data[0]["question_id"] == "test_q1"

    def test_get_data_load_has_sentence_breakdown(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)
        data = task.get_data(load=True)
        entry = data[0]
        assert "sentences" in entry
        assert "middle_start_idx" in entry
        assert "middle_end_idx" in entry
        assert "target_num_sentences" in entry
        assert "char_budget" in entry
        assert entry["middle_start_idx"] < entry["middle_end_idx"]

    def test_get_activations_returns_false(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")
        assert task.get_activations(load=False) is False
        assert task.get_activations(load=True) is None


# ---------------------------------------------------------------------------
# CompressedCotTask — prepare_for_monitor()
# ---------------------------------------------------------------------------

class TestCompressedCotPrepareForMonitor:
    def test_returns_row_dicts(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)
        rows = task.prepare_for_monitor()
        assert isinstance(rows, list)
        assert len(rows) >= 1

        row = rows[0]
        assert "question" in row
        assert "full_cot" in row
        assert "sentences" in row
        assert "middle_start_idx" in row
        assert "middle_end_idx" in row
        assert "target_num_sentences" in row
        assert "char_budget" in row
        assert "question_type" in row

    def test_multiple_choice_has_choices(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)
        rows = task.prepare_for_monitor()
        row = rows[0]
        assert "choices" in row
        assert "correct_answer" in row

    def test_data_slice_filtering(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        from src2.data_slice import DataSlice

        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)

        # Should return nothing for a non-existent ID
        rows = task.prepare_for_monitor(DataSlice.from_ids(["nonexistent"]))
        assert len(rows) == 0

        # Should return data for the correct ID
        rows = task.prepare_for_monitor(DataSlice.from_ids(["test_q1"]))
        assert len(rows) >= 1

    def test_raises_when_no_data(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "empty")
        with pytest.raises(RuntimeError, match="No data found"):
            task.prepare_for_monitor()


# ---------------------------------------------------------------------------
# CompressedCotTask — evaluate()
# ---------------------------------------------------------------------------

class TestCompressedCotEvaluate:
    def test_identical_distributions(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")

        dist = {"A": 0.7, "B": 0.2, "C": 0.05, "D": 0.05}
        result = task.evaluate([dist], [dist])
        assert result["kl_divergence"] < 0.01
        assert result["js_divergence"] < 0.01
        assert result["answer_agreement"] == 1.0

    def test_different_distributions(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")

        pred = {"A": 0.9, "B": 0.1}
        gt = {"A": 0.1, "B": 0.9}
        result = task.evaluate([pred], [gt])
        assert result["kl_divergence"] > 0.5
        assert result["js_divergence"] > 0.1
        assert result["answer_agreement"] == 0.0

    def test_empty_data(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")
        result = task.evaluate([], [])
        assert result["kl_divergence"] == float("inf")
        assert result["js_divergence"] == 1.0
        assert result["answer_agreement"] == 0.0

    def test_multiple_rows(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")

        preds = [
            {"A": 0.8, "B": 0.2},
            {"A": 0.3, "B": 0.7},
        ]
        gts = [
            {"A": 0.8, "B": 0.2},
            {"A": 0.3, "B": 0.7},
        ]
        result = task.evaluate(preds, gts)
        assert result["kl_divergence"] < 0.01
        assert result["answer_agreement"] == 1.0

    def test_none_predictions_skipped(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")

        preds = [None, {"A": 0.8, "B": 0.2}]
        gts = [{"A": 0.5, "B": 0.5}, {"A": 0.8, "B": 0.2}]
        result = task.evaluate(preds, gts)
        # Only the non-None pair should be evaluated
        assert result["answer_agreement"] == 1.0


# ---------------------------------------------------------------------------
# CompressedCotTask — run_data() with mocked verification data
# ---------------------------------------------------------------------------

class TestCompressedCotRunData:
    def test_run_data_loads_and_saves(self, tmp_dir):
        from src2.tasks import CompressedCotTask

        data_dir = tmp_dir / "cot_data"
        verification_dir = data_dir.parent / "verification_rollouts"
        _write_verification_rollout(verification_dir, "test_q1", n_sentences=20)

        task = CompressedCotTask(model="test/model", data_dir=data_dir)
        result = task.run_data("test_q1", rollout_idx=0, verbose=False)

        assert result is not None
        assert result["question_id"] == "test_q1"
        assert result["num_sentences"] == 20
        assert result["middle_start_idx"] == 5  # 20 // 4
        assert result["middle_end_idx"] == 15   # 3 * 20 // 4
        assert result["middle_num_sentences"] == 10
        assert result["compression_factor"] == 10
        assert result["target_num_sentences"] >= 1
        assert result["char_budget"] > 0

        # Verify files written
        assert task.get_data(load=False) is True

    def test_run_data_returns_none_for_missing_question(self, tmp_dir):
        from src2.tasks import CompressedCotTask
        task = CompressedCotTask(model="test/model", data_dir=tmp_dir / "cot")
        result = task.run_data("nonexistent_question", verbose=False)
        assert result is None

    def test_run_data_custom_compression_factor(self, tmp_dir):
        from src2.tasks import CompressedCotTask

        data_dir = tmp_dir / "cot_data"
        verification_dir = data_dir.parent / "verification_rollouts"
        _write_verification_rollout(verification_dir, "test_q1", n_sentences=40)

        task = CompressedCotTask(
            model="test/model", compression_factor=5, data_dir=data_dir,
        )
        result = task.run_data("test_q1", rollout_idx=0, verbose=False)

        assert result is not None
        assert result["compression_factor"] == 5
        # middle = 40//4 to 3*40//4 = 10 to 30 = 20 sentences
        # target = 20 // 5 = 4
        assert result["target_num_sentences"] == 4


# ---------------------------------------------------------------------------
# SentenceSelectionPrompt
# ---------------------------------------------------------------------------

class TestSentenceSelectionPrompt:
    def test_format_returns_string(self):
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt
        prompt = SentenceSelectionPrompt()

        row = {
            "question": "What is 2+2?",
            "sentences": [f"Sentence {i}." for i in range(20)],
            "middle_start_idx": 5,
            "middle_end_idx": 15,
            "target_num_sentences": 2,
            "char_budget": 100,
        }
        result = prompt.format(row)
        assert isinstance(result, str)
        assert "What is 2+2?" in result
        assert "[5]" in result
        assert "[14]" in result
        assert "2 most important" in result
        assert "100 characters" in result

    def test_parse_response_valid_json(self):
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt
        prompt = SentenceSelectionPrompt()

        indices = prompt.parse_response("[5, 8, 12]")
        assert indices == [5, 8, 12]

    def test_parse_response_with_surrounding_text(self):
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt
        prompt = SentenceSelectionPrompt()

        indices = prompt.parse_response("Here are the indices:\n[10, 15, 20]\nDone.")
        assert indices == [10, 15, 20]

    def test_parse_response_invalid(self):
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt
        prompt = SentenceSelectionPrompt()

        assert prompt.parse_response("no json here") is None
        assert prompt.parse_response("") is None

    def test_parse_response_non_integer_array(self):
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt
        prompt = SentenceSelectionPrompt()

        # Array with strings should return None
        assert prompt.parse_response('["a", "b"]') is None

    def test_name(self):
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt
        prompt = SentenceSelectionPrompt()
        assert prompt.name == "sentence_selection"


# ---------------------------------------------------------------------------
# SummaryCompressionPrompt
# ---------------------------------------------------------------------------

class TestSummaryCompressionPrompt:
    def test_format_returns_string(self):
        from src2.tasks.compressed_cot.prompts import SummaryCompressionPrompt
        prompt = SummaryCompressionPrompt()

        row = {
            "question": "What is 2+2?",
            "sentences": [f"Sentence {i}." for i in range(20)],
            "middle_start_idx": 5,
            "middle_end_idx": 15,
            "char_budget": 200,
        }
        result = prompt.format(row)
        assert isinstance(result, str)
        assert "What is 2+2?" in result
        assert "200 characters" in result
        assert "Sentence 5." in result   # Middle section includes sentence 5
        assert "Sentence 0." in result   # First quarter context

    def test_parse_response_returns_text(self):
        from src2.tasks.compressed_cot.prompts import SummaryCompressionPrompt
        prompt = SummaryCompressionPrompt()

        result = prompt.parse_response("This is the compressed reasoning.")
        assert result == "This is the compressed reasoning."

    def test_parse_response_strips_whitespace(self):
        from src2.tasks.compressed_cot.prompts import SummaryCompressionPrompt
        prompt = SummaryCompressionPrompt()

        result = prompt.parse_response("  some text  \n\n")
        assert result == "some text"

    def test_parse_response_empty(self):
        from src2.tasks.compressed_cot.prompts import SummaryCompressionPrompt
        prompt = SummaryCompressionPrompt()

        assert prompt.parse_response("") is None
        assert prompt.parse_response("   ") is None

    def test_name(self):
        from src2.tasks.compressed_cot.prompts import SummaryCompressionPrompt
        prompt = SummaryCompressionPrompt()
        assert prompt.name == "summary_compression"


# ---------------------------------------------------------------------------
# CompressedCotTask — LlmMonitor integration (mocked openai)
# ---------------------------------------------------------------------------

class TestCompressedCotWithLlmMonitor:
    def test_sentence_selection_monitor_workflow(self, tmp_dir):
        """Full workflow: task → prepare_for_monitor → LlmMonitor with SentenceSelectionPrompt."""
        from src2.tasks import CompressedCotTask
        from src2.methods import LlmMonitor
        from src2.tasks.compressed_cot.prompts import SentenceSelectionPrompt

        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)

        prompt = SentenceSelectionPrompt()
        monitor = LlmMonitor(
            prompt=prompt, model="test/model",
            max_workers=1, api_key="fake-key",
        )

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[6, 8, 10, 12]"
        mock_client.chat.completions.create.return_value = mock_response
        monitor.client = mock_client

        monitor.set_task(task)
        folder = monitor.get_folder()

        rows = task.prepare_for_monitor()
        results = monitor.infer(rows, verbose=False)

        assert len(results) >= 1
        assert all("monitor_prediction" in r for r in results)
        assert results[0]["monitor_prediction"] == [6, 8, 10, 12]
        assert (folder / "results.jsonl").exists()

    def test_summary_compression_monitor_workflow(self, tmp_dir):
        """Full workflow: task → prepare_for_monitor → LlmMonitor with SummaryCompressionPrompt."""
        from src2.tasks import CompressedCotTask
        from src2.methods import LlmMonitor
        from src2.tasks.compressed_cot.prompts import SummaryCompressionPrompt

        data_dir = tmp_dir / "cot_data"
        write_compressed_cot_data(data_dir, question_id="test_q1", n_sentences=20)
        task = CompressedCotTask(model="test/model", data_dir=data_dir)

        prompt = SummaryCompressionPrompt()
        monitor = LlmMonitor(
            prompt=prompt, model="test/model",
            max_workers=1, api_key="fake-key",
        )

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Compressed reasoning text here."
        mock_client.chat.completions.create.return_value = mock_response
        monitor.client = mock_client

        monitor.set_task(task)
        folder = monitor.get_folder()

        rows = task.prepare_for_monitor()
        results = monitor.infer(rows, verbose=False)

        assert len(results) >= 1
        assert all("monitor_prediction" in r for r in results)
        assert results[0]["monitor_prediction"] == "Compressed reasoning text here."
        assert (folder / "results.jsonl").exists()


# ---------------------------------------------------------------------------
# CompressedCotTask — internal helpers
# ---------------------------------------------------------------------------

class TestCompressedCotHelpers:
    def test_user_msg_multiple_choice(self, tmp_dir):
        from src2.tasks.compressed_cot.task import CompressedCotTask
        from src2.utils.questions import GPQAQuestion

        q = GPQAQuestion(
            id="q1", question="What?",
            choices=["A opt", "B opt", "C opt", "D opt"],
            correct_answer="B", correct_index=1,
        )
        msg = CompressedCotTask._user_msg(q)
        assert "What?" in msg
        assert "A. A opt" in msg
        assert "Answer with just the letter" in msg

    def test_user_msg_binary_judge(self, tmp_dir):
        from src2.tasks.compressed_cot.task import CompressedCotTask
        from src2.utils.questions import BinaryJudgeQuestion

        q = BinaryJudgeQuestion(
            id="q1", question="Is this bad?",
            judge_prompt="Judge: {response}", bad_outcome="YES",
        )
        msg = CompressedCotTask._user_msg(q)
        assert msg == "Is this bad?"

    def test_question_from_summary_mc(self, tmp_dir):
        from src2.tasks.compressed_cot.task import CompressedCotTask
        from src2.utils.questions import GPQAQuestion

        summary = {
            "question_id": "q1",
            "question": "What?",
            "question_type": "multiple_choice",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "B",
        }
        q = CompressedCotTask._question_from_summary(summary)
        assert isinstance(q, GPQAQuestion)
        assert q.correct_answer == "B"

    def test_question_from_summary_binary(self, tmp_dir):
        from src2.tasks.compressed_cot.task import CompressedCotTask
        from src2.utils.questions import BinaryJudgeQuestion

        summary = {
            "question_id": "q1",
            "question": "Is this bad?",
            "question_type": "binary_judge",
            "judge_prompt": "Judge: {response}",
            "bad_outcome": "YES",
        }
        q = CompressedCotTask._question_from_summary(summary)
        assert isinstance(q, BinaryJudgeQuestion)
        assert q.bad_outcome == "YES"


# ---------------------------------------------------------------------------
# Helpers — write synthetic verification rollout + compressed_cot data
# ---------------------------------------------------------------------------

def _write_verification_rollout(
    verification_dir: Path,
    question_id: str,
    n_sentences: int = 20,
):
    """Write a synthetic verification rollout for testing run_data()."""
    run_dir = verification_dir / question_id / "20260101_000000"
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    # Build a CoT with distinct sentences
    sentences = [f"Step {i}: reasoning about part {i}." for i in range(n_sentences)]
    thinking = " ".join(sentences)

    rollout = {
        "rollout_idx": 0,
        "full_prompt": "prompt text",
        "thinking": thinking,
        "answer": "B",
        "full_response": f"<think>{thinking}</think>B",
    }
    with open(rollouts_dir / "rollout_000.json", "w") as f:
        json.dump(rollout, f)

    summary = {
        "question_id": question_id,
        "question": "What is 2+2?\n\nA. 3\nB. 4\nC. 5\nD. 6",
        "question_type": "multiple_choice",
        "choices": ["3", "4", "5", "6"],
        "correct_answer": "B",
        "total_rollouts": 1,
        "valid_rollouts": 1,
        "answer_counts": {"B": 1},
        "most_common_answer": "B",
        "most_common_count": 1,
        "agreement_rate": 1.0,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f)
