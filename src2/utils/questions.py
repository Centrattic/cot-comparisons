"""
Question data types for forced-response tasks.

Moved from src2/tasks/forced_response/data_loader.py to be shared across tasks.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd


@dataclass
class GPQAQuestion:
    """A single GPQA question (multiple choice)."""

    id: str
    question: str
    choices: List[str]
    correct_answer: str  # The letter (A, B, C, D)
    correct_index: int  # 0-indexed
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    question_type: str = "multiple_choice"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPQAQuestion":
        return cls(**data)


@dataclass
class BinaryJudgeQuestion:
    """A question that requires an LLM judge to classify the response."""

    id: str
    question: str
    judge_prompt: str  # Prompt template for the judge, with {response} placeholder
    bad_outcome: str  # What the judge returns if the model misbehaves (e.g., "YES")
    subject: Optional[str] = None
    question_type: str = "binary_judge"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BinaryJudgeQuestion":
        return cls(**data)


# Sample GPQA questions for initial testing
SAMPLE_GPQA_QUESTIONS = [
    GPQAQuestion(
        id="gpqa_sample_001",
        question="A spin-1/2 particle is in a state |ψ⟩ = (1/√2)|↑⟩ + (1/√2)|↓⟩. If we measure the spin along the z-axis, what is the probability of finding the particle in the spin-up state?",
        choices=["0", "1/4", "1/2", "1"],
        correct_answer="C",
        correct_index=2,
        subject="physics",
        difficulty="graduate",
    ),
    GPQAQuestion(
        id="gpqa_sample_002",
        question="In organic chemistry, what is the major product when 2-butene undergoes hydroboration-oxidation?",
        choices=["1-butanol", "2-butanol", "2-butanone", "butanal"],
        correct_answer="B",
        correct_index=1,
        subject="chemistry",
        difficulty="graduate",
    ),
    GPQAQuestion(
        id="gpqa_sample_003",
        question="Consider a continuous function f: [0,1] → [0,1]. By Brouwer's fixed point theorem, f must have at least one fixed point. If f(0) = 1 and f(1) = 0, and f is strictly decreasing, how many fixed points does f have?",
        choices=["Exactly 0", "Exactly 1", "Exactly 2", "Infinitely many"],
        correct_answer="B",
        correct_index=1,
        subject="mathematics",
        difficulty="graduate",
    ),
    GPQAQuestion(
        id="gpqa_sample_004",
        question="In the context of computational complexity theory, which of the following statements about the class BPP is correct?",
        choices=[
            "BPP is known to be strictly contained in NP",
            "BPP is known to equal P",
            "BPP is contained in the polynomial hierarchy (specifically, in Σ₂ ∩ Π₂)",
            "BPP is known to contain NP",
        ],
        correct_answer="C",
        correct_index=2,
        subject="computer_science",
        difficulty="graduate",
    ),
    GPQAQuestion(
        id="gpqa_sample_005",
        question="What is the product of the reaction between benzaldehyde and acetone in the presence of a base catalyst?",
        choices=[
            "Benzyl alcohol",
            "4-phenyl-3-buten-2-one (benzylideneacetone)",
            "Diphenylmethane",
            "Benzophenone",
        ],
        correct_answer="B",
        correct_index=1,
        subject="chemistry",
        difficulty="graduate",
    ),
]


# Type alias for any question type
Question = Union[GPQAQuestion, BinaryJudgeQuestion]


def load_custom_questions(
    questions_file: Optional[Path] = None,
) -> List[Question]:
    """
    Load custom questions from questions.json.

    Supports both multiple_choice and binary_judge question types.
    """
    if questions_file is None:
        questions_file = Path("questions.json")

    if not questions_file.exists():
        return []

    with open(questions_file) as f:
        data = json.load(f)

    questions: List[Question] = []
    for item in data:
        question_type = item.get("type", "multiple_choice")

        if question_type == "binary_judge":
            questions.append(
                BinaryJudgeQuestion(
                    id=item["id"],
                    question=item["question"],
                    judge_prompt=item["judge_prompt"],
                    bad_outcome=item["bad_outcome"],
                    subject=item.get("subject"),
                )
            )
        else:
            correct_answer = item["correct_answer"]
            correct_index = ord(correct_answer) - ord("A")
            questions.append(
                GPQAQuestion(
                    id=item["id"],
                    question=item["question"],
                    choices=item["choices"],
                    correct_answer=correct_answer,
                    correct_index=correct_index,
                    subject=item.get("subject"),
                    difficulty=item.get("difficulty"),
                )
            )

    return questions


def load_gpqa_questions(
    data_dir: Optional[Path] = None,
    use_samples: bool = True,
    max_questions: Optional[int] = None,
) -> List[GPQAQuestion]:
    """
    Load GPQA questions.

    Args:
        data_dir: Directory containing GPQA data files
        use_samples: If True, use sample questions (for initial testing)
        max_questions: Maximum number of questions to load
    """
    if use_samples:
        questions = SAMPLE_GPQA_QUESTIONS.copy()
    else:
        if data_dir is None:
            data_dir = (
                Path(__file__).parent.parent.parent
                / "data"
                / "forced_response"
                / "gpqa"
            )

        questions_file = data_dir / "questions.json"
        if questions_file.exists():
            with open(questions_file) as f:
                data = json.load(f)
            questions = [GPQAQuestion.from_dict(q) for q in data]
        else:
            raise FileNotFoundError(
                f"GPQA questions file not found at {questions_file}. "
                "Use use_samples=True for sample questions or provide the data file."
            )

    if max_questions is not None:
        questions = questions[:max_questions]

    return questions


def save_gpqa_questions(
    questions: List[GPQAQuestion],
    data_dir: Optional[Path] = None,
) -> Path:
    """Save GPQA questions to JSON file."""
    if data_dir is None:
        data_dir = (
            Path(__file__).parent.parent.parent / "data" / "forced_response" / "gpqa"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    questions_file = data_dir / "questions.json"

    with open(questions_file, "w") as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)

    return questions_file


def load_gpqa_from_huggingface(
    subset: str = "gpqa_diamond",
    max_questions: Optional[int] = None,
) -> List[GPQAQuestion]:
    """
    Load GPQA questions from HuggingFace datasets.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    dataset = load_dataset("Idavidrein/gpqa", subset, split="train")

    questions = []
    for i, item in enumerate(dataset):
        if max_questions is not None and i >= max_questions:
            break

        choices = [
            item.get("Incorrect Answer 1", ""),
            item.get("Incorrect Answer 2", ""),
            item.get("Incorrect Answer 3", ""),
            item.get("Correct Answer", ""),
        ]

        import hashlib

        seed = int(hashlib.md5(item["Question"].encode()).hexdigest()[:8], 16)
        import random

        rng = random.Random(seed)

        choice_pairs = [(c, i == 3) for i, c in enumerate(choices)]
        rng.shuffle(choice_pairs)

        shuffled_choices = [c for c, _ in choice_pairs]
        correct_idx = next(
            i for i, (_, is_correct) in enumerate(choice_pairs) if is_correct
        )
        correct_letter = chr(ord("A") + correct_idx)

        questions.append(
            GPQAQuestion(
                id=f"gpqa_{subset}_{i:04d}",
                question=item["Question"],
                choices=shuffled_choices,
                correct_answer=correct_letter,
                correct_index=correct_idx,
                subject=item.get("Subdomain", None),
                difficulty=subset,
            )
        )

    return questions
