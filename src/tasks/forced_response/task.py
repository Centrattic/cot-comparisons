"""
Forced Response Task

Analyzes chain-of-thought by forcing the model to answer at different points.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pandas as pd

from ..base import BaseTask
from .data_loader import GPQAQuestion, load_gpqa_questions


class ForcedResponseTask(BaseTask):
    """
    Task for analyzing forced responses at different CoT points.

    This task:
    1. Runs verification: 50 rollouts to find questions with >80% agreement
    2. For verified questions, forces the model to answer after each sentence
    3. (Future) Resamples answer distribution from each point
    """

    def __init__(self, model: str = "moonshotai/kimi-k2-thinking"):
        super().__init__("forced_response")
        self.model = model
        self.task_dir = Path(__file__).parent.parent.parent.parent / "data" / "forced_response"
        self.verification_dir = self.task_dir / "verification"
        self.forcing_dir = self.task_dir / "forcing"
        self.monitor_dir = self.task_dir / "monitor"
        self.resampling_dir = self.task_dir / "resampling"

        # Ensure directories exist
        for d in [self.verification_dir, self.forcing_dir, self.monitor_dir, self.resampling_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_question_dir(self, question_id: str, mode: str) -> Path:
        """
        Get directory for a specific question in a specific mode.

        Args:
            question_id: The question ID
            mode: "verification", "forcing", or "resampling"

        Returns:
            Path to the question directory
        """
        if mode == "verification":
            base_dir = self.verification_dir
        elif mode == "forcing":
            base_dir = self.forcing_dir
        elif mode == "monitor":
            base_dir = self.monitor_dir
        elif mode == "resampling":
            base_dir = self.resampling_dir
        else:
            raise ValueError(f"Unknown mode: {mode}")

        question_dir = base_dir / question_id
        question_dir.mkdir(parents=True, exist_ok=True)
        return question_dir

    def get_ground_truth(self, row) -> str:
        """Get the correct answer letter for a question."""
        return row.get("correct_answer", "")

    def evaluate(
        self,
        predictions: List[str],
        ground_truth: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of predicted answer letters
            ground_truth: List of correct answer letters

        Returns:
            Dictionary with accuracy metric
        """
        if not predictions or not ground_truth:
            return {"accuracy": 0.0}

        correct = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p.upper() == g.upper()
        )
        return {"accuracy": correct / len(predictions)}

    def save_verification_result(
        self,
        question: GPQAQuestion,
        rollouts: List[Dict[str, Any]],
    ) -> Path:
        """
        Save verification results for a question.

        Args:
            question: The GPQA question
            rollouts: List of rollout results

        Returns:
            Path to the saved summary file
        """
        question_dir = self.get_question_dir(question.id, "verification")
        rollouts_dir = question_dir / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Save individual rollouts
        for i, rollout in enumerate(rollouts):
            rollout_path = rollouts_dir / f"rollout_{i:03d}.json"
            with open(rollout_path, "w") as f:
                json.dump(rollout, f, indent=2)

        # Compute summary statistics
        answers = [r.get("answer", "").upper() for r in rollouts if r.get("answer")]
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        total_valid = len(valid_answers)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)
        agreement_rate = most_common[1] / total_valid if total_valid > 0 else 0

        summary = {
            "question_id": question.id,
            "question": question.question,
            "choices": question.choices,
            "correct_answer": question.correct_answer,
            "total_rollouts": len(rollouts),
            "valid_rollouts": total_valid,
            "answer_counts": answer_counts,
            "most_common_answer": most_common[0],
            "most_common_count": most_common[1],
            "agreement_rate": agreement_rate,
            "is_correct": most_common[0] == question.correct_answer,
            "meets_threshold": agreement_rate >= 0.8,
        }

        summary_path = question_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_forcing_result(
        self,
        question: GPQAQuestion,
        sentence_idx: int,
        partial_cot: str,
        force_results: List[Dict[str, Any]],
        rollout_idx: int = 0,
    ) -> Path:
        """
        Save forcing results for a specific sentence in the CoT.

        Args:
            question: The GPQA question
            sentence_idx: Index of the sentence in the CoT
            partial_cot: The partial CoT up to this point
            force_results: List of force attempt results
            rollout_idx: Index of the source rollout being forced

        Returns:
            Path to the saved results file
        """
        question_dir = self.get_question_dir(question.id, "forcing")
        rollout_dir = question_dir / f"rollout_{rollout_idx:03d}"
        sentence_dir = rollout_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Save individual force attempts
        for i, result in enumerate(force_results):
            force_path = sentence_dir / f"force_{i:03d}.json"
            with open(force_path, "w") as f:
                json.dump(result, f, indent=2)

        # Compute summary for this sentence
        answers = [r.get("answer", "").upper() for r in force_results]
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]
        valid_single_token = [
            r for r in force_results
            if r.get("is_valid_single_token", False)
        ]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        summary = {
            "question_id": question.id,
            "sentence_idx": sentence_idx,
            "partial_cot": partial_cot,
            "total_attempts": len(force_results),
            "valid_single_token": len(valid_single_token),
            "answer_counts": answer_counts,
        }

        summary_path = sentence_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def load_verification_summary(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Load verification summary for a question."""
        summary_path = self.verification_dir / question_id / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return None

    def get_verified_questions(self, threshold: float = 0.8) -> List[str]:
        """
        Get list of question IDs that meet the verification threshold.

        Args:
            threshold: Minimum agreement rate required

        Returns:
            List of question IDs meeting the threshold
        """
        verified = []
        if self.verification_dir.exists():
            for question_dir in self.verification_dir.iterdir():
                if question_dir.is_dir():
                    summary = self.load_verification_summary(question_dir.name)
                    if summary and summary.get("agreement_rate", 0) >= threshold:
                        verified.append(question_dir.name)
        return verified

    def save_forcing_summary(
        self,
        question: GPQAQuestion,
        source_rollout_idx: int,
        all_sentence_results: List[Dict[str, Any]],
        source_cot: str = "",
    ) -> Path:
        """
        Save overall forcing summary for a question inside the rollout directory.

        Args:
            question: The GPQA question
            source_rollout_idx: Index of the rollout used for forcing
            all_sentence_results: List of summaries for each sentence
            source_cot: The full source chain of thought from the rollout

        Returns:
            Path to the saved summary file
        """
        question_dir = self.get_question_dir(question.id, "forcing")
        rollout_dir = question_dir / f"rollout_{source_rollout_idx:03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "question_id": question.id,
            "source_rollout_idx": source_rollout_idx,
            "correct_answer": question.correct_answer,
            "num_sentences": len(all_sentence_results),
            "source_cot": source_cot,
            "sentence_summaries": all_sentence_results,
        }

        summary_path = rollout_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_monitor_result(
        self,
        question: GPQAQuestion,
        sentence_idx: int,
        partial_cot: str,
        force_results: List[Dict[str, Any]],
        rollout_idx: int = 0,
    ) -> Path:
        """
        Save monitor results for a specific sentence in the CoT.

        Args:
            question: The GPQA question
            sentence_idx: Index of the sentence in the CoT
            partial_cot: The partial CoT up to this point
            force_results: List of monitor attempt results
            rollout_idx: Index of the source rollout being monitored

        Returns:
            Path to the saved results file
        """
        question_dir = self.get_question_dir(question.id, "monitor")
        rollout_dir = question_dir / f"rollout_{rollout_idx:03d}"
        sentence_dir = rollout_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Save individual monitor attempts
        for i, result in enumerate(force_results):
            force_path = sentence_dir / f"force_{i:03d}.json"
            with open(force_path, "w") as f:
                json.dump(result, f, indent=2)

        # Compute summary for this sentence
        answers = [r.get("answer", "").upper() for r in force_results]
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]
        valid_single_token = [
            r for r in force_results
            if r.get("is_valid_single_token", False)
        ]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        summary = {
            "question_id": question.id,
            "sentence_idx": sentence_idx,
            "partial_cot": partial_cot,
            "total_attempts": len(force_results),
            "valid_single_token": len(valid_single_token),
            "answer_counts": answer_counts,
        }

        summary_path = sentence_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_monitor_summary(
        self,
        question: GPQAQuestion,
        source_rollout_idx: int,
        all_sentence_results: List[Dict[str, Any]],
        source_cot: str = "",
    ) -> Path:
        """
        Save overall monitor summary for a question inside the rollout directory.

        Args:
            question: The GPQA question
            source_rollout_idx: Index of the rollout used for monitoring
            all_sentence_results: List of summaries for each sentence
            source_cot: The full source chain of thought from the rollout

        Returns:
            Path to the saved summary file
        """
        question_dir = self.get_question_dir(question.id, "monitor")
        rollout_dir = question_dir / f"rollout_{source_rollout_idx:03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "question_id": question.id,
            "source_rollout_idx": source_rollout_idx,
            "correct_answer": question.correct_answer,
            "num_sentences": len(all_sentence_results),
            "source_cot": source_cot,
            "sentence_summaries": all_sentence_results,
        }

        summary_path = rollout_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_resampling_result(
        self,
        question: GPQAQuestion,
        sentence_idx: int,
        forced_prefix: str,
        resample_results: List[Dict[str, Any]],
    ) -> Path:
        """
        Save resampling results for a specific sentence in the CoT.

        Args:
            question: The GPQA question
            sentence_idx: Index of the sentence in the CoT
            forced_prefix: The forced prefix used for resampling
            resample_results: List of resample attempt results

        Returns:
            Path to the saved results file
        """
        question_dir = self.get_question_dir(question.id, "resampling")
        sentence_dir = question_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Save individual resample attempts
        for i, result in enumerate(resample_results):
            resample_path = sentence_dir / f"resample_{i:03d}.json"
            with open(resample_path, "w") as f:
                json.dump(result, f, indent=2)

        # Compute summary for this sentence
        answers = [r.get("answer", "").upper() for r in resample_results]
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        total_valid = len(valid_answers)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)

        summary = {
            "question_id": question.id,
            "sentence_idx": sentence_idx,
            "forced_prefix": forced_prefix,
            "total_resamples": len(resample_results),
            "valid_answers": total_valid,
            "answer_counts": answer_counts,
            "most_common": most_common[0],
            "most_common_count": most_common[1],
            "agreement_rate": most_common[1] / total_valid if total_valid > 0 else 0,
        }

        summary_path = sentence_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_resampling_summary(
        self,
        question: GPQAQuestion,
        source_rollout_idx: int,
        all_sentence_results: List[Dict[str, Any]],
    ) -> Path:
        """
        Save overall resampling summary for a question.

        Args:
            question: The GPQA question
            source_rollout_idx: Index of the rollout used for resampling
            all_sentence_results: List of summaries for each sentence

        Returns:
            Path to the saved summary file
        """
        question_dir = self.get_question_dir(question.id, "resampling")

        summary = {
            "question_id": question.id,
            "source_rollout_idx": source_rollout_idx,
            "correct_answer": question.correct_answer,
            "num_sentences": len(all_sentence_results),
            "sentence_summaries": all_sentence_results,
        }

        summary_path = question_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path
