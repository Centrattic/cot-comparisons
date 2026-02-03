"""Tests for ThoughtAnchors method."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ThoughtAnchors — construction
# ---------------------------------------------------------------------------

class TestThoughtAnchorsRequired:
    def test_missing_model(self):
        from src2.methods import ThoughtAnchors
        with pytest.raises(TypeError, match="model"):
            ThoughtAnchors()

    def test_default_params(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")
        assert ta.name == "thought_anchors"
        assert ta.sampling_schedule == [10, 20, 30, 50, 70, 100]
        assert ta.confidence_level == 0.95
        assert ta.bootstrap_reps == 1000
        assert ta.max_workers == 300

    def test_custom_params(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(
            model="test/model",
            sampling_schedule=[5, 10],
            confidence_level=0.90,
            bootstrap_reps=500,
            max_workers=50,
        )
        assert ta.sampling_schedule == [5, 10]
        assert ta.confidence_level == 0.90
        assert ta.bootstrap_reps == 500
        assert ta.max_workers == 50


# ---------------------------------------------------------------------------
# ThoughtAnchors — get_config()
# ---------------------------------------------------------------------------

class TestThoughtAnchorsConfig:
    def test_config_has_required_keys(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")
        config = ta.get_config()
        assert config["name"] == "thought_anchors"
        assert config["model"] == "test/model"
        assert "sampling_schedule" in config
        assert "confidence_level" in config

    def test_config_excludes_private(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")
        config = ta.get_config()
        for key in config:
            assert not key.startswith("_"), f"Private key {key} leaked"


# ---------------------------------------------------------------------------
# ThoughtAnchors — get_folder() and method_config.json
# ---------------------------------------------------------------------------

class TestThoughtAnchorsGetFolder:
    def test_creates_method_config(self, tmp_dir):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")

        task = MagicMock()
        task.data_dir = tmp_dir / "task_data"
        task.data_dir.mkdir(parents=True, exist_ok=True)
        task.get_config.return_value = {"name": "compressed_cot"}

        ta.set_task(task)
        folder = ta.get_folder()

        config_path = folder / "method_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)
        assert "method" in config
        assert "task" in config
        assert config["method"]["name"] == "thought_anchors"

    def test_raises_without_set_task(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")
        with pytest.raises(RuntimeError, match="set_task"):
            ta.get_folder()


# ---------------------------------------------------------------------------
# ThoughtAnchors — _build_question_obj()
# ---------------------------------------------------------------------------

class TestBuildQuestionObj:
    def test_multiple_choice(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        row = {
            "question_id": "q1",
            "question": "What is 2+2?",
            "question_type": "multiple_choice",
            "choices": ["3", "4", "5", "6"],
            "correct_answer": "B",
        }
        q = ThoughtAnchors._build_question_obj(row)
        assert isinstance(q, GPQAQuestion)
        assert q.correct_answer == "B"
        assert q.choices == ["3", "4", "5", "6"]

    def test_binary_judge(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import BinaryJudgeQuestion

        row = {
            "question_id": "q1",
            "question": "Is this bad?",
            "question_type": "binary_judge",
            "bad_outcome": "YES",
        }
        q = ThoughtAnchors._build_question_obj(row)
        assert isinstance(q, BinaryJudgeQuestion)
        assert q.bad_outcome == "YES"


# ---------------------------------------------------------------------------
# ThoughtAnchors — _compute_accuracy()
# ---------------------------------------------------------------------------

class TestComputeAccuracy:
    def test_mc_all_correct(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model")
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)
        assert ta._compute_accuracy(["B", "B", "B"], q) == 1.0

    def test_mc_none_correct(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model")
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)
        assert ta._compute_accuracy(["A", "C", "D"], q) == 0.0

    def test_mc_partial(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model")
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)
        assert ta._compute_accuracy(["B", "A", "B", "C"], q) == pytest.approx(0.5)

    def test_binary_judge(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import BinaryJudgeQuestion

        ta = ThoughtAnchors(model="test/model")
        q = BinaryJudgeQuestion(id="q", question="?", judge_prompt="", bad_outcome="YES")
        # 2 out of 3 are NOT bad → accuracy = 2/3
        assert ta._compute_accuracy(["NO", "YES", "NO"], q) == pytest.approx(2 / 3)

    def test_empty_answers(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model")
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)
        assert ta._compute_accuracy([], q) == 0.0


# ---------------------------------------------------------------------------
# ThoughtAnchors — _bootstrap_ci()
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_all_correct_ci(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model", bootstrap_reps=500)
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)

        # All answers are correct, baseline is 1.0
        # So importance = 1.0 - 1.0 = 0.0 with tight CI around 0
        answers = ["B"] * 50
        ci_low, ci_high = ta._bootstrap_ci(answers, q, baseline_accuracy=1.0)
        assert ci_low <= 0.0 <= ci_high  # CI should include 0

    def test_all_wrong_high_importance(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model", bootstrap_reps=500)
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)

        # All answers wrong, baseline was 1.0
        # importance = 1.0 - 0.0 = 1.0
        answers = ["A"] * 50
        ci_low, ci_high = ta._bootstrap_ci(answers, q, baseline_accuracy=1.0)
        assert ci_low > 0.5  # Should be clearly important

    def test_returns_tuple(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        ta = ThoughtAnchors(model="test/model", bootstrap_reps=100)
        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)

        answers = ["B", "A", "B", "A", "B", "A", "B", "B", "A", "B"]
        ci_low, ci_high = ta._bootstrap_ci(answers, q, baseline_accuracy=0.8)
        assert isinstance(ci_low, float)
        assert isinstance(ci_high, float)
        assert ci_low <= ci_high


# ---------------------------------------------------------------------------
# ThoughtAnchors — _extract_answer()
# ---------------------------------------------------------------------------

class TestThoughtAnchorsExtractAnswer:
    def test_extract_mc_answer(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "some thinking</think>B"

        answer, cot, text = ThoughtAnchors._extract_answer([1, 2, 3], tokenizer, q)
        assert answer == "B"

    def test_extract_binary_answer(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import BinaryJudgeQuestion

        q = BinaryJudgeQuestion(id="q", question="?", judge_prompt="", bad_outcome="YES")

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "some thinking</think>YES"

        answer, cot, text = ThoughtAnchors._extract_answer([1, 2, 3], tokenizer, q)
        assert answer == "YES"

    def test_no_think_tag(self):
        from src2.methods.thought_anchors import ThoughtAnchors
        from src2.utils.questions import GPQAQuestion

        q = GPQAQuestion(id="q", question="?", choices=["a", "b", "c", "d"],
                         correct_answer="B", correct_index=1)

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "some text without think tag"

        answer, cot, text = ThoughtAnchors._extract_answer([1, 2, 3], tokenizer, q)
        assert answer == ""  # No </think> → can't extract answer


# ---------------------------------------------------------------------------
# ThoughtAnchors — _save_results()
# ---------------------------------------------------------------------------

class TestThoughtAnchorsSaveResults:
    def test_saves_jsonl_and_summary(self, tmp_dir):
        from src2.methods import ThoughtAnchors

        ta = ThoughtAnchors(model="test/model")

        task = MagicMock()
        task.data_dir = tmp_dir / "task_data"
        task.data_dir.mkdir(parents=True, exist_ok=True)
        task.get_config.return_value = {"name": "mock_task"}

        ta.set_task(task)
        folder = ta.get_folder()

        results = [
            {
                "question_id": "q1",
                "importance_scores": {"5": 0.3, "6": 0.1},
                "selected_indices": [5],
                "compressed_cot": "First. Step 5. Last.",
                "compressed_middle": "Step 5.",
                "samples_used": {"5": 30, "6": 10},
                "baseline_accuracy": 0.8,
                "num_sentences_selected": 1,
                "compressed_char_count": 7,
            }
        ]
        ta._save_results(results)

        assert (folder / "results.jsonl").exists()
        assert (folder / "summary.json").exists()

        with open(folder / "results.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["question_id"] == "q1"
        assert parsed["selected_indices"] == [5]

        with open(folder / "summary.json") as f:
            summary = json.load(f)
        assert len(summary) == 1
        assert summary[0]["question_id"] == "q1"
        assert summary[0]["num_sentences_selected"] == 1


# ---------------------------------------------------------------------------
# ThoughtAnchors — infer() raises without setup
# ---------------------------------------------------------------------------

class TestThoughtAnchorsInferGuards:
    def test_infer_without_set_task_raises(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")
        with pytest.raises(RuntimeError, match="set_task"):
            ta.infer([{"question": "test"}])

    def test_infer_without_get_folder_raises(self):
        from src2.methods import ThoughtAnchors
        ta = ThoughtAnchors(model="test/model")

        task = MagicMock()
        task.data_dir = Path("/tmp/test")
        task.data_dir.mkdir(parents=True, exist_ok=True)
        task.get_config.return_value = {"name": "mock"}

        ta.set_task(task)
        # Don't call get_folder()
        with pytest.raises(RuntimeError):
            ta.infer([{"question": "test"}])


# ---------------------------------------------------------------------------
# ThoughtAnchors — _make_serializable()
# ---------------------------------------------------------------------------

class TestMakeSerializable:
    def test_primitives(self):
        from src2.methods.thought_anchors import _make_serializable
        assert _make_serializable("hello") == "hello"
        assert _make_serializable(42) == 42
        assert _make_serializable(3.14) == 3.14
        assert _make_serializable(True) is True
        assert _make_serializable(None) is None

    def test_list(self):
        from src2.methods.thought_anchors import _make_serializable
        assert _make_serializable([1, "a", None]) == [1, "a", None]

    def test_dict(self):
        from src2.methods.thought_anchors import _make_serializable
        assert _make_serializable({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_numpy_scalar(self):
        from src2.methods.thought_anchors import _make_serializable
        val = np.float64(3.14)
        result = _make_serializable(val)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_nested(self):
        from src2.methods.thought_anchors import _make_serializable
        data = {"scores": [np.float64(0.5), np.int64(3)], "name": "test"}
        result = _make_serializable(data)
        assert result == {"scores": [0.5, 3], "name": "test"}

    def test_non_serializable_becomes_str(self):
        from src2.methods.thought_anchors import _make_serializable

        class Foo:
            def __str__(self):
                return "foo_object"

        result = _make_serializable(Foo())
        assert result == "foo_object"
