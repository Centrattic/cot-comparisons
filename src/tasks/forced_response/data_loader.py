"""
Data loader for GPQA (Graduate-Level Google-Proof Q&A) dataset.

GPQA is a challenging multiple-choice dataset designed to test expert-level reasoning.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class GPQAQuestion:
    """A single GPQA question."""
    id: str
    question: str
    choices: List[str]
    correct_answer: str  # The letter (A, B, C, D)
    correct_index: int   # 0-indexed
    subject: Optional[str] = None
    difficulty: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPQAQuestion":
        return cls(**data)


# Sample GPQA questions for initial testing
# These are representative questions - in production, load from the full dataset
SAMPLE_GPQA_QUESTIONS = [
    GPQAQuestion(
        id="gpqa_sample_001",
        question="A spin-1/2 particle is in a state |ψ⟩ = (1/√2)|↑⟩ + (1/√2)|↓⟩. If we measure the spin along the z-axis, what is the probability of finding the particle in the spin-up state?",
        choices=[
            "0",
            "1/4",
            "1/2",
            "1"
        ],
        correct_answer="C",
        correct_index=2,
        subject="physics",
        difficulty="graduate",
    ),
    GPQAQuestion(
        id="gpqa_sample_002",
        question="In organic chemistry, what is the major product when 2-butene undergoes hydroboration-oxidation?",
        choices=[
            "1-butanol",
            "2-butanol",
            "2-butanone",
            "butanal"
        ],
        correct_answer="B",
        correct_index=1,
        subject="chemistry",
        difficulty="graduate",
    ),
    GPQAQuestion(
        id="gpqa_sample_003",
        question="Consider a continuous function f: [0,1] → [0,1]. By Brouwer's fixed point theorem, f must have at least one fixed point. If f(0) = 1 and f(1) = 0, and f is strictly decreasing, how many fixed points does f have?",
        choices=[
            "Exactly 0",
            "Exactly 1",
            "Exactly 2",
            "Infinitely many"
        ],
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
            "BPP is known to contain NP"
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
            "Benzophenone"
        ],
        correct_answer="B",
        correct_index=1,
        subject="chemistry",
        difficulty="graduate",
    ),
]


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

    Returns:
        List of GPQAQuestion objects
    """
    if use_samples:
        questions = SAMPLE_GPQA_QUESTIONS.copy()
    else:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "forced_response" / "gpqa"

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
    """
    Save GPQA questions to JSON file.

    Args:
        questions: List of GPQAQuestion objects
        data_dir: Directory to save to

    Returns:
        Path to saved file
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "forced_response" / "gpqa"

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

    Args:
        subset: GPQA subset to load ("gpqa_diamond", "gpqa_extended", "gpqa_main")
        max_questions: Maximum number of questions to load

    Returns:
        List of GPQAQuestion objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install with: pip install datasets"
        )

    # Load from HuggingFace
    dataset = load_dataset("Idavidrein/gpqa", subset, split="train")

    questions = []
    for i, item in enumerate(dataset):
        if max_questions is not None and i >= max_questions:
            break

        # Extract choices and find correct answer
        choices = [
            item.get("Incorrect Answer 1", ""),
            item.get("Incorrect Answer 2", ""),
            item.get("Incorrect Answer 3", ""),
            item.get("Correct Answer", ""),
        ]

        # Shuffle choices deterministically based on question
        import hashlib
        seed = int(hashlib.md5(item["Question"].encode()).hexdigest()[:8], 16)
        import random
        rng = random.Random(seed)

        # Create (choice, is_correct) pairs
        choice_pairs = [(c, i == 3) for i, c in enumerate(choices)]
        rng.shuffle(choice_pairs)

        # Find correct index after shuffling
        shuffled_choices = [c for c, _ in choice_pairs]
        correct_idx = next(i for i, (_, is_correct) in enumerate(choice_pairs) if is_correct)
        correct_letter = chr(ord('A') + correct_idx)

        questions.append(GPQAQuestion(
            id=f"gpqa_{subset}_{i:04d}",
            question=item["Question"],
            choices=shuffled_choices,
            correct_answer=correct_letter,
            correct_index=correct_idx,
            subject=item.get("Subdomain", None),
            difficulty=subset,
        ))

    return questions
