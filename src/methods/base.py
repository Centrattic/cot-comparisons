from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseMethod(ABC):
    """
    Base class for all CoT analysis methods.

    Methods receive rollouts containing paths to data files (cot_content, response,
    activations) and produce predictions that can be evaluated against ground truth.
    """

    def __init__(self, name: str):
        """
        Initialize a method.

        Args:
            name: Method identifier, used as column name in results CSV.
        """
        self.name = name

    @abstractmethod
    def predict(self, rollout: Dict[str, Path]) -> Any:
        """
        Make a prediction given a single CoT rollout.

        Args:
            rollout: Dict containing paths:
                - 'cot_content': Path to CoT text file
                - 'response': Path to response file
                - 'activations': Path to activations file (optional)

        Returns:
            Prediction value. Type depends on the task:
            - float (0-1) for binary prediction tasks
            - set of labels for multi-label tasks
            - task-specific for intervention tasks
        """
        pass

    def predict_batch(self, rollouts: List[Dict[str, Path]]) -> List[Any]:
        """
        Batch prediction over multiple rollouts.

        Override this method for efficiency if your method benefits from
        batching (e.g., batched API calls or GPU inference).

        Args:
            rollouts: List of rollout dicts with paths.

        Returns:
            List of predictions, one per rollout.
        """
        return [self.predict(r) for r in rollouts]

    @staticmethod
    def load_text(path: Path) -> str:
        """Helper to load text content from a path."""
        with open(path, "r") as f:
            return f.read()


class MonitorMethod(BaseMethod):
    """
    Methods that analyze CoT text only (no model internals needed).

    Examples:
    - LLM-based monitors that read the CoT and make judgments
    - Regex/keyword-based detectors
    - Text classifiers
    """

    def __init__(self, name: str):
        super().__init__(name)

    def predict(self, rollout: Dict[str, Path]) -> Any:
        """Load CoT text from path and delegate to analyze_cot."""
        cot_path = rollout.get("cot_content")

        if cot_path is None:
            raise ValueError("Rollout missing 'cot_content' path")

        cot_text = self.load_text(cot_path)
        return self.analyze_cot(cot_text)

    @abstractmethod
    def analyze_cot(self, cot_text: str) -> Any:
        """
        Analyze chain-of-thought text and return a prediction.

        Args:
            cot_text: The chain-of-thought content as a string.

        Returns:
            Prediction value appropriate for the task.
        """
        pass


class ProbeMethod(BaseMethod):
    """
    Methods that use trained probes on model activations.

    Examples:
    - Linear probes on hidden states
    - Attention-based probes (rolling mean attention probe from GDM paper)
    - MLP probes
    """

    def __init__(self, name: str):
        super().__init__(name)

    def predict(self, rollout: Dict[str, Path]) -> Any:
        """Load activations and CoT, delegate to analyze_with_probe."""
        cot_path = rollout.get("cot_content")
        activation_path = rollout.get("activations")

        if not cot_path or not activation_path:
            raise ValueError("Rollout missing 'cot_content' or 'activations' path")

        cot_text = self.load_text(cot_path)

        return self.analyze_with_probe(cot_text, activation_path)

    @abstractmethod
    def analyze_with_probe(self, cot_text: str, activation_path: Optional[Path]) -> Any:
        """
        Analyze using a probe on model activations.

        Args:
            cot_text: The chain-of-thought content.
            activation_path: Path to saved activations file.

        Returns:
            Prediction value appropriate for the task.
        """
        pass


class ActivationOracleMethod(BaseMethod):
    """
    Methods using activation oracles for analysis.

    These methods work directly with activations without necessarily
    needing the text content.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def predict(self, rollout: Dict[str, Path]) -> Any:
        """Extract activation path and delegate to analyze_activations."""
        activation_path = rollout.get("activations")
        return self.analyze_activations(activation_path)

    @abstractmethod
    def analyze_activations(self, activation_path: Optional[Path]) -> Any:
        """
        Analyze activations directly.

        Args:
            activation_path: Path to saved activations file.

        Returns:
            Prediction value appropriate for the task.
        """
        pass


class GlobalAnalysisMethod(BaseMethod):
    """
    Methods that analyze multiple rollouts together.

    Examples:
    - Algorithm graphs that compare multiple rollouts
    - Clustering-based analysis
    - Multi-rollout LLM analysis
    """

    def __init__(self, name: str):
        super().__init__(name)

    def predict(self, rollout: Dict[str, Path]) -> Any:
        """Single-rollout prediction not supported for global methods."""
        raise NotImplementedError(
            "GlobalAnalysisMethod requires multiple rollouts. Use predict_batch()."
        )

    @abstractmethod
    def predict_batch(self, rollouts: List[Dict[str, Path]]) -> List[Any]:
        """
        Analyze all rollouts together and produce predictions.

        Args:
            rollouts: List of all rollout dicts with paths.

        Returns:
            List of predictions, one per rollout.
        """
        pass
