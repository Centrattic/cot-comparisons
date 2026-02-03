"""Shared fixtures for src2 tests."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure src2 is importable (no pyproject.toml / setup.py in this repo)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src2.utils.questions import GPQAQuestion


# ---------------------------------------------------------------------------
# Basic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Fresh temp directory per test."""
    return tmp_path


@pytest.fixture
def sample_activations():
    """Random activations for probe training: (10, 128)."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 128).astype(np.float32)


@pytest.fixture
def sample_gpqa_question():
    """One GPQAQuestion instance."""
    return GPQAQuestion(
        id="test_q1",
        question="What is 2+2?",
        choices=["3", "4", "5", "6"],
        correct_answer="B",
        correct_index=1,
        subject="math",
        difficulty="easy",
    )


# ---------------------------------------------------------------------------
# Mock task fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_task(tmp_dir):
    """Minimal ScruplesTask with mocked openai client (enough to construct)."""
    from src2.tasks import ScruplesTask

    task = ScruplesTask(
        subject_model="openrouter/test/model",
        variant="first_person",
        data_dir=tmp_dir / "scruples_data",
        api_key="fake-key-for-testing",
    )
    # Replace client with a mock so no real API calls happen
    task.client = MagicMock()
    return task


# ---------------------------------------------------------------------------
# Helpers for writing synthetic data to disk
# ---------------------------------------------------------------------------

def write_scruples_csvs(data_dir: Path, variant: str = "first_person",
                        n_anecdotes: int = 3, n_runs: int = 2):
    """Write synthetic results_<variant>.csv and prompts_<variant>.csv."""
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = data_dir / "runs" / "2026-01-01_00-00-00"

    results_rows = []
    prompts_rows = []

    for i in range(n_anecdotes):
        aid = f"anecdote_{i}"
        for arm in ("control", "intervention"):
            for run_idx in range(n_runs):
                is_syco = arm == "intervention" and run_idx == 0
                run_path = f"runs/2026-01-01_00-00-00/{aid}/{arm}_{run_idx}.json"
                results_rows.append({
                    "anecdote_id": aid,
                    "run_idx": run_idx,
                    "arm": arm,
                    "variant": variant,
                    "answer": "A" if is_syco else "B",
                    "is_sycophantic": is_syco,
                    "run_path": run_path,
                })
                # Write corresponding run JSON
                json_path = data_dir / run_path
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_data = {
                    "anecdote_id": aid,
                    "run_idx": run_idx,
                    "arm": arm,
                    "variant": variant,
                    "prompt": f"Prompt for {aid}",
                    "thinking": f"Thinking for {aid} {arm} {run_idx}",
                    "answer": "A" if is_syco else "B",
                    "full_response": f"Response for {aid}",
                    "is_sycophantic": is_syco,
                    "author_is_wrong": True,
                }
                with open(json_path, "w") as f:
                    json.dump(json_data, f)

        switch_rate = 0.5 if i % 2 == 0 else 0.0
        prompts_rows.append({
            "anecdote_id": aid,
            "title": f"Title {i}",
            "text": f"Text for anecdote {i}",
            "label": "WRONG" if i % 2 == 0 else "RIGHT",
            "consensus_ratio": 0.9,
            "author_is_wrong": True,
            "variant": variant,
            "num_control_runs": n_runs,
            "control_sycophantic_count": 0,
            "control_sycophancy_rate": 0.0,
            "num_intervention_runs": n_runs,
            "intervention_sycophantic_count": 1,
            "intervention_sycophancy_rate": 0.5,
            "switch_rate": switch_rate,
            "effect_classification": "moderate",
            "total_votes": 10,
            "label_scores": None,
        })

    pd.DataFrame(results_rows).to_csv(
        data_dir / f"results_{variant}.csv", index=False
    )
    pd.DataFrame(prompts_rows).to_csv(
        data_dir / f"prompts_{variant}.csv", index=False
    )
    return data_dir


def write_forcing_data(data_dir: Path, question_id: str = "test_q1",
                       n_sentences: int = 3, n_forces: int = 2):
    """Write synthetic forcing summary.json + force_*.json files."""
    forcing_dir = data_dir / "forcing"

    # Create a run directory structure
    run_dir = forcing_dir / question_id / "rollout_000" / "20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    sentence_summaries = []
    for si in range(n_sentences):
        sentence_dir = run_dir / f"sentence_{si:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        answer_counts = {"A": n_forces // 2, "B": n_forces - n_forces // 2}
        for fi in range(n_forces):
            partial_cot = f"Thinking up to sentence {si}."
            full_prompt = (
                "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
                f"<|im_user|>user<|im_middle|>What is 2+2?<|im_end|>"
                f"<|im_assistant|>assistant<|im_middle|><think>{partial_cot}"
            )
            force_data = {
                "sentence_idx": si,
                "force_idx": fi,
                "partial_cot": partial_cot,
                "continued_cot": f"Continued thinking after sentence {si}.",
                "raw_tokens": [1, 2, 3],
                "raw_response": f"<think>thinking</think>{'A' if fi == 0 else 'B'}",
                "answer": "A" if fi == 0 else "B",
                "full_prompt": full_prompt,
            }
            with open(sentence_dir / f"force_{fi:03d}.json", "w") as f:
                json.dump(force_data, f)

        # Sentence summary
        summary = {
            "question_id": question_id,
            "question_type": "multiple_choice",
            "sentence_idx": si,
            "partial_cot": f"Thinking up to sentence {si}.",
            "total_attempts": n_forces,
            "valid_answers": n_forces,
            "answer_counts": answer_counts,
        }
        with open(sentence_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        sentence_summaries.append({
            "sentence_idx": si,
            "partial_cot_length": len(f"Thinking up to sentence {si}."),
            "total_forces": n_forces,
            "valid_answers": n_forces,
            "answer_counts": answer_counts,
            "most_common": "B",
        })

    # Overall summary
    overall_summary = {
        "question_id": question_id,
        "question_type": "multiple_choice",
        "source_rollout_idx": 0,
        "num_sentences": n_sentences,
        "source_cot": "Full source CoT text here.",
        "sentence_summaries": sentence_summaries,
        "correct_answer": "B",
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f)

    # Also write config.json expected by create_run_dir
    config = {
        "model": "test-model",
        "question_id": question_id,
        "timestamp": "2026-01-01T00:00:00",
        "run_type": "forcing",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)

    return data_dir


def write_compressed_cot_data(data_dir: Path, question_id: str = "test_q1",
                              n_sentences: int = 20):
    """Write synthetic compressed_cot task_data.json for testing."""
    sentences = [f"Step {i}: reasoning about part {i}." for i in range(n_sentences)]
    full_cot = " ".join(sentences)

    n = len(sentences)
    middle_start = n // 4
    middle_end = 3 * n // 4
    middle_sentences = sentences[middle_start:middle_end]
    compression_factor = 10
    target_num_sentences = max(1, len(middle_sentences) // compression_factor)
    middle_char_count = sum(len(s) for s in middle_sentences)
    char_budget = int(middle_char_count / compression_factor * 1.5)

    task_data = {
        "question_id": question_id,
        "question": "What is 2+2?\n\nA. 3\nB. 4\nC. 5\nD. 6\n\nAnswer with just the letter (A, B, C, or D).",
        "question_type": "multiple_choice",
        "rollout_idx": 0,
        "full_cot": full_cot,
        "sentences": sentences,
        "num_sentences": n,
        "middle_start_idx": middle_start,
        "middle_end_idx": middle_end,
        "middle_num_sentences": len(middle_sentences),
        "middle_char_count": middle_char_count,
        "compression_factor": compression_factor,
        "target_num_sentences": target_num_sentences,
        "char_budget": char_budget,
        "model": "test/model",
        "correct_answer": "B",
        "choices": ["3", "4", "5", "6"],
    }

    run_dir = data_dir / question_id / "rollout_000" / "20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "task_data.json", "w") as f:
        json.dump(task_data, f, indent=2)

    config = {
        "question_id": question_id,
        "model": "test/model",
        "timestamp": "2026-01-01T00:00:00",
        "compression_factor": compression_factor,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return data_dir
