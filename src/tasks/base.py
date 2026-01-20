import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class BaseTask(ABC):
    """Base class for all CoT analysis tasks."""

    name: str  # Task identifier (e.g., "blackmail")

    def __init__(self, name: str):
        """
        Initialize a task.

        Args:
            task_dir: Path to the task's data directory. If None, defaults to
                      data/{name}/ relative to the project root.
        """
        self.name = name
        # Default: project_root/tasks/{name}/
        task_dir = Path(__file__).parent.parent.parent / "data" / self.name

        self.task_dir = Path(task_dir)
        self._data: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load the task's CSV data from data.csv."""

        csv_path = self.task_dir / "data.csv"

        if csv_path.exists():
            self._data = pd.read_csv(csv_path)
        else:
            self._data = pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        """The task's data as a DataFrame."""
        if self._data is None:
            self._load_data()
        return self._data

    @abstractmethod
    def get_ground_truth(self, row: pd.Series) -> Any:
        """
        Extract ground truth label from a data row.

        Args:
            row: A row from the task's DataFrame.

        Returns:
            The ground truth value for this example.
        """
        pass

    def get_rollout(self, row: pd.Series) -> Dict[str, Path]:
        """
        Get paths to rollout data for a data row.

        Args:
            row: A row from the task's DataFrame.

        Returns:
            Dict with paths:
                - 'cot_content': Path to the CoT content file
                - 'response': Path to the response file
                - 'activations': Path to activations file (if present)
        """
        rollout: Dict[str, Path] = {}

        if "cot_path" in row and pd.notna(row["cot_path"]):
            rollout["cot_content"] = self.task_dir / row["cot_path"]

        if "response_path" in row and pd.notna(row["response_path"]):
            rollout["response"] = self.task_dir / row["response_path"]

        if "activation_path" in row and pd.notna(row["activation_path"]):
            rollout["activations"] = self.task_dir / row["activation_path"]

        return rollout

    @abstractmethod
    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """
        Compute task-specific evaluation metrics.

        Args:
            predictions: List of predictions from a method.
            ground_truth: List of ground truth values.

        Returns:
            Dict mapping metric names to values, e.g., {'auc': 0.85, 'accuracy': 0.80}
        """
        pass

    def save_results(self, method_name: str, predictions: List[Any]) -> None:
        """
        Append method predictions as a new column to data.csv.

        Args:
            method_name: Name of the method (becomes column name).
            predictions: List of predictions, one per row.
        """
        self._data[method_name] = predictions
        self._data.to_csv(self.task_dir / "data.csv", index=False)


class BinaryPredictionTask(BaseTask):
    """
    Task where ground truth is binary (0/1) and methods predict probability.

    Examples: blackmail prediction, answer correctness prediction.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def get_ground_truth(self, row: pd.Series) -> int:
        """Extract binary ground truth (0 or 1)."""
        return int(row["ground_truth"])

    def evaluate(
        self, predictions: List[float], ground_truth: List[int]
    ) -> Dict[str, float]:
        """
        Compute binary classification metrics.

        Args:
            predictions: Predicted probabilities (0.0 to 1.0).
            ground_truth: Binary labels (0 or 1).

        Returns:
            Dict with 'auc' and 'accuracy'.
        """
        from sklearn.metrics import accuracy_score, roc_auc_score

        binary_preds = [1 if p > 0.5 else 0 for p in predictions]

        metrics = {
            "accuracy": accuracy_score(ground_truth, binary_preds),
        }

        # AUC requires both classes present
        if len(set(ground_truth)) > 1:
            metrics["auc"] = roc_auc_score(ground_truth, predictions)
        else:
            metrics["auc"] = float("nan")

        return metrics


class InterventionTask(BaseTask):
    """
    Task involving interventions on model behavior.

    Example: Scruples task where black-box monitors perform poorly.
    """
    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def get_intervention_result(self, row: pd.Series) -> Any:
        """
        Get the result of the intervention for evaluation.

        Args:
            row: A row from the task's DataFrame.

        Returns:
            The intervention result for this example.
        """
        pass


class MultiLabelTask(BaseTask):
    """
    Task where we predict multiple labels per example.

    Example: Process task where we need to identify ALL methods covered.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def get_ground_truth(self, row: pd.Series) -> set:
        """
        Extract ground truth as a set of labels that are comma-separated
        """
        gt = row["ground_truth"]

        return set(label.strip() for label in gt.split(","))

    def evaluate(
        self, predictions: List[set], ground_truth: List[set]
    ) -> Dict[str, float]:
        """
        Compute multi-label metrics.

        Args:
            predictions: List of predicted label sets.
            ground_truth: List of ground truth label sets.

        Returns:
            Dict with 'exact_match', 'precision', 'recall', 'f1'.
        """
        n = len(ground_truth)
        if n == 0:
            return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        exact_matches = sum(p == g for p, g in zip(predictions, ground_truth))

        # Micro-averaged precision/recall
        total_predicted = sum(len(p) for p in predictions)
        total_true = sum(len(g) for g in ground_truth)
        total_correct = sum(len(p & g) for p, g in zip(predictions, ground_truth))

        precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        recall = total_correct / total_true if total_true > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "exact_match": exact_matches / n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
