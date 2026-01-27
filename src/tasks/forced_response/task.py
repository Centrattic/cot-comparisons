"""
Forced Response Task

Analyzes chain-of-thought by forcing the model to answer at different points.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pandas as pd

from ..base import BaseTask
from .data_loader import GPQAQuestion, BinaryJudgeQuestion, Question, load_gpqa_questions


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
        self.monitor_forcing_dir = self.task_dir / "monitor_forcing"
        self.monitor_resampling_dir = self.task_dir / "monitor_resampling"
        self.resampling_dir = self.task_dir / "resampling"

        # Ensure directories exist
        for d in [self.verification_dir, self.forcing_dir, self.monitor_forcing_dir,
                  self.monitor_resampling_dir, self.resampling_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_question_dir(self, question_id: str, mode: str) -> Path:
        """
        Get directory for a specific question in a specific mode.

        Args:
            question_id: The question ID
            mode: "verification", "forcing", "monitor_forcing", "monitor_resampling", or "resampling"

        Returns:
            Path to the question directory
        """
        mode_dirs = {
            "verification": self.verification_dir,
            "forcing": self.forcing_dir,
            "monitor_forcing": self.monitor_forcing_dir,
            "monitor_resampling": self.monitor_resampling_dir,
            "resampling": self.resampling_dir,
        }
        if mode not in mode_dirs:
            raise ValueError(f"Unknown mode: {mode}")

        question_dir = mode_dirs[mode] / question_id
        question_dir.mkdir(parents=True, exist_ok=True)
        return question_dir

    def create_run_dir(self, mode: str, question_id: str, rollout_idx: int, config: dict) -> Path:
        """
        Create a timestamped run directory with config.json.

        Args:
            mode: Run type ("forcing", "resampling", "monitor_forcing", "monitor_resampling")
            question_id: The question ID
            rollout_idx: Index of the source rollout
            config: Configuration dict to save as config.json

        Returns:
            Path to the created timestamped run directory
        """
        base = self.get_question_dir(question_id, mode)
        rollout_dir = base / f"rollout_{rollout_idx:03d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = rollout_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        config["timestamp"] = datetime.now().isoformat()
        config["run_type"] = mode
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return run_dir

    def create_verification_run_dir(self, question_id: str, config: dict) -> Path:
        """
        Create a timestamped run directory for verification with config.json.

        Args:
            question_id: The question ID
            config: Configuration dict to save as config.json

        Returns:
            Path to the created timestamped run directory
        """
        base = self.get_question_dir(question_id, "verification")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        config["timestamp"] = datetime.now().isoformat()
        config["run_type"] = "verification"
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return run_dir

    @staticmethod
    def get_latest_run_dir(rollout_dir: Path) -> Optional[Path]:
        """
        Get the latest timestamped run directory inside a rollout dir.

        Falls back to the rollout_dir itself if no timestamped subdirs exist
        (for backward compatibility with migrated data).

        Args:
            rollout_dir: Path to the rollout directory

        Returns:
            Path to the latest run directory, or None if nothing exists
        """
        if not rollout_dir.exists():
            return None
        # Look for timestamped subdirs (YYYYMMDD_HHMMSS format)
        timestamped = sorted(
            [d for d in rollout_dir.iterdir()
             if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
            reverse=True,
        )
        if timestamped:
            return timestamped[0]
        # Fallback: if summary.json exists directly in rollout_dir (pre-migration)
        if (rollout_dir / "summary.json").exists():
            return rollout_dir
        return None

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
        question: Question,
        rollouts: List[Dict[str, Any]],
        run_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save verification results for a question.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            rollouts: List of rollout results
            run_dir: Timestamped run directory (if None, falls back to legacy path)

        Returns:
            Path to the saved summary file
        """
        if run_dir:
            save_dir = run_dir
        else:
            save_dir = self.get_question_dir(question.id, "verification")
        rollouts_dir = save_dir / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        # Save individual rollouts
        for i, rollout in enumerate(rollouts):
            rollout_path = rollouts_dir / f"rollout_{i:03d}.json"
            with open(rollout_path, "w") as f:
                json.dump(rollout, f, indent=2)

        # Compute summary statistics - valid answers depend on question type
        answers = [r.get("answer", "").upper() for r in rollouts if r.get("answer")]
        if isinstance(question, BinaryJudgeQuestion):
            valid_answers = [a for a in answers if a in ["YES", "NO"]]
        else:
            valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        total_valid = len(valid_answers)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)
        agreement_rate = most_common[1] / total_valid if total_valid > 0 else 0

        # Build summary - different fields for different question types
        summary = {
            "question_id": question.id,
            "question": question.question,
            "question_type": question.question_type,
            "total_rollouts": len(rollouts),
            "valid_rollouts": total_valid,
            "answer_counts": answer_counts,
            "most_common_answer": most_common[0],
            "most_common_count": most_common[1],
            "agreement_rate": agreement_rate,
            "meets_threshold": agreement_rate >= 0.8,
        }

        if isinstance(question, BinaryJudgeQuestion):
            summary["judge_prompt"] = question.judge_prompt
            summary["bad_outcome"] = question.bad_outcome
            summary["bad_outcome_rate"] = answer_counts.get(question.bad_outcome.upper(), 0) / total_valid if total_valid > 0 else 0
            summary["subject"] = question.subject
        else:
            summary["choices"] = question.choices
            summary["correct_answer"] = question.correct_answer
            summary["is_correct"] = most_common[0] == question.correct_answer

        summary_path = save_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_forcing_result(
        self,
        question: Question,
        sentence_idx: int,
        partial_cot: str,
        force_results: List[Dict[str, Any]],
        rollout_idx: int = 0,
        run_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save forcing results for a specific sentence in the CoT.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            sentence_idx: Index of the sentence in the CoT
            partial_cot: The partial CoT up to this point
            force_results: List of force attempt results
            rollout_idx: Index of the source rollout being forced
            run_dir: Timestamped run directory (if None, falls back to legacy path)

        Returns:
            Path to the saved results file
        """
        if run_dir:
            sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        else:
            question_dir = self.get_question_dir(question.id, "forcing")
            rollout_dir = question_dir / f"rollout_{rollout_idx:03d}"
            sentence_dir = rollout_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Save individual force attempts
        for i, result in enumerate(force_results):
            force_path = sentence_dir / f"force_{i:03d}.json"
            with open(force_path, "w") as f:
                json.dump(result, f, indent=2)

        # Compute summary for this sentence - valid answers depend on question type
        answers = [r.get("answer", "").upper() for r in force_results]
        if isinstance(question, BinaryJudgeQuestion):
            valid_answers = [a for a in answers if a in ["YES", "NO"]]
        else:
            valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        summary = {
            "question_id": question.id,
            "question_type": question.question_type,
            "sentence_idx": sentence_idx,
            "partial_cot": partial_cot,
            "total_attempts": len(force_results),
            "valid_answers": len(valid_answers),
            "answer_counts": answer_counts,
        }

        summary_path = sentence_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def get_latest_verification_dir(self, question_id: str) -> Optional[Path]:
        """
        Get the latest timestamped verification directory for a question.

        Falls back to question_dir itself if no timestamped subdirs exist
        (for backward compatibility with pre-migration data).

        Args:
            question_id: The question ID

        Returns:
            Path to the latest verification run directory, or None if nothing exists
        """
        question_dir = self.verification_dir / question_id
        if not question_dir.exists():
            return None
        # Look for timestamped subdirs (YYYYMMDD_HHMMSS format)
        timestamped = sorted(
            [d for d in question_dir.iterdir()
             if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
            reverse=True,
        )
        if timestamped:
            return timestamped[0]
        # Fallback: if summary.json exists directly in question_dir (pre-migration)
        if (question_dir / "summary.json").exists():
            return question_dir
        return None

    def load_verification_summary(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Load verification summary for a question (from latest timestamped run)."""
        run_dir = self.get_latest_verification_dir(question_id)
        if run_dir:
            summary_path = run_dir / "summary.json"
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
        question: Question,
        source_rollout_idx: int,
        all_sentence_results: List[Dict[str, Any]],
        source_cot: str = "",
        run_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save overall forcing summary for a question.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            source_rollout_idx: Index of the rollout used for forcing
            all_sentence_results: List of summaries for each sentence
            source_cot: The full source chain of thought from the rollout
            run_dir: Timestamped run directory (if None, falls back to legacy path)

        Returns:
            Path to the saved summary file
        """
        if run_dir:
            save_dir = run_dir
        else:
            question_dir = self.get_question_dir(question.id, "forcing")
            save_dir = question_dir / f"rollout_{source_rollout_idx:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "question_id": question.id,
            "question_type": question.question_type,
            "source_rollout_idx": source_rollout_idx,
            "num_sentences": len(all_sentence_results),
            "source_cot": source_cot,
            "sentence_summaries": all_sentence_results,
        }

        if isinstance(question, BinaryJudgeQuestion):
            summary["bad_outcome"] = question.bad_outcome
        else:
            summary["correct_answer"] = question.correct_answer

        summary_path = save_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_monitor_result(
        self,
        question: Question,
        sentence_idx: int,
        partial_cot: str,
        force_results: List[Dict[str, Any]],
        run_dir: Path,
    ) -> Path:
        """
        Save monitor results for a specific sentence in the CoT.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            sentence_idx: Index of the sentence in the CoT
            partial_cot: The partial CoT up to this point
            force_results: List of monitor attempt results
            run_dir: Timestamped run directory

        Returns:
            Path to the saved results file
        """
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Save individual monitor attempts
        for i, result in enumerate(force_results):
            force_path = sentence_dir / f"force_{i:03d}.json"
            with open(force_path, "w") as f:
                json.dump(result, f, indent=2)

        # Compute summary for this sentence - valid answers depend on question type
        answers = [r.get("answer", "").upper() for r in force_results]
        if isinstance(question, BinaryJudgeQuestion):
            valid_answers = [a for a in answers if a in ["YES", "NO"]]
        else:
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
            "question_type": question.question_type,
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
        question: Question,
        source_rollout_idx: int,
        all_sentence_results: List[Dict[str, Any]],
        source_cot: str = "",
        run_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save overall monitor summary.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            source_rollout_idx: Index of the rollout used for monitoring
            all_sentence_results: List of summaries for each sentence
            source_cot: The full source chain of thought from the rollout
            run_dir: Timestamped run directory

        Returns:
            Path to the saved summary file
        """
        save_dir = run_dir if run_dir else Path(".")

        summary = {
            "question_id": question.id,
            "question_type": question.question_type,
            "source_rollout_idx": source_rollout_idx,
            "num_sentences": len(all_sentence_results),
            "source_cot": source_cot,
            "sentence_summaries": all_sentence_results,
        }

        if isinstance(question, BinaryJudgeQuestion):
            summary["bad_outcome"] = question.bad_outcome
        else:
            summary["correct_answer"] = question.correct_answer

        summary_path = save_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def save_resampling_result(
        self,
        question: Question,
        sentence_idx: int,
        forced_prefix: str,
        resample_results: List[Dict[str, Any]],
        rollout_idx: int = 0,
        run_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save resampling results for a specific sentence in the CoT.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            sentence_idx: Index of the sentence in the CoT
            forced_prefix: The forced prefix used for resampling
            resample_results: List of resample attempt results
            rollout_idx: Index of the source rollout being resampled
            run_dir: Timestamped run directory (if None, falls back to legacy path)

        Returns:
            Path to the saved results file
        """
        if run_dir:
            sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        else:
            question_dir = self.get_question_dir(question.id, "resampling")
            rollout_dir = question_dir / f"rollout_{rollout_idx:03d}"
            sentence_dir = rollout_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Save individual resample attempts
        for i, result in enumerate(resample_results):
            resample_path = sentence_dir / f"resample_{i:03d}.json"
            with open(resample_path, "w") as f:
                json.dump(result, f, indent=2)

        # Compute summary for this sentence - valid answers depend on question type
        answers = [r.get("answer", "").upper() for r in resample_results]
        if isinstance(question, BinaryJudgeQuestion):
            valid_answers = [a for a in answers if a in ["YES", "NO"]]
        else:
            valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        total_valid = len(valid_answers)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)

        summary = {
            "question_id": question.id,
            "question_type": question.question_type,
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
        question: Question,
        source_rollout_idx: int,
        all_sentence_results: List[Dict[str, Any]],
        run_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save overall resampling summary for a question.

        Args:
            question: The question (GPQAQuestion or BinaryJudgeQuestion)
            source_rollout_idx: Index of the rollout used for resampling
            all_sentence_results: List of summaries for each sentence
            run_dir: Timestamped run directory (if None, falls back to legacy path)

        Returns:
            Path to the saved summary file
        """
        if run_dir:
            save_dir = run_dir
        else:
            question_dir = self.get_question_dir(question.id, "resampling")
            save_dir = question_dir / f"rollout_{source_rollout_idx:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "question_id": question.id,
            "question_type": question.question_type,
            "source_rollout_idx": source_rollout_idx,
            "num_sentences": len(all_sentence_results),
            "sentence_summaries": all_sentence_results,
        }

        if isinstance(question, BinaryJudgeQuestion):
            summary["bad_outcome"] = question.bad_outcome
        else:
            summary["correct_answer"] = question.correct_answer

        summary_path = save_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path
