"""
Base task ABC for src2.

Tasks are data providers: they generate rollouts/activations and serve them to methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class BaseTask(ABC):
    """
    Abstract base for all tasks.

    Subclasses must implement:
      - run_data()         : generate rollouts + activations
      - get_data()         : check existence (load=False) or return DataFrames (load=True)
      - get_activations()  : check existence (load=False) or return paths/data (load=True)
      - evaluate()         : score predictions against ground truth
    """

    name: str
    data_dir: Path

    def __init__(self, name: str, data_dir: Optional[Path] = None):
        self.name = name
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            # Default: project_root/data/{name}/
            self.data_dir = Path(__file__).parent.parent.parent / "data" / name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_config(self) -> Dict[str, Any]:
        """Return a dict of this task's configuration for serialization.

        Default implementation returns all public, JSON-serializable attributes.
        Subclasses can override for custom behavior.
        """
        config = {"name": self.name, "data_dir": str(self.data_dir)}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key in ("name", "data_dir"):
                continue
            if isinstance(value, Path):
                config[key] = str(value)
            elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
                config[key] = value
        return config

    @abstractmethod
    def run_data(self, **kwargs) -> None:
        """Generate rollouts and activations. Saves to self.data_dir."""
        ...

    @abstractmethod
    def get_data(self, load: bool = False) -> Union[bool, Optional[Dict[str, pd.DataFrame]]]:
        """
        If load=False: return True if data exists, False otherwise.
        If load=True:  return dict of DataFrames (e.g. {"results": ..., "prompts": ...})
                       or None if data doesn't exist.
        """
        ...

    @abstractmethod
    def get_activations(self, load: bool = False) -> Union[bool, Optional[Any]]:
        """
        If load=False: return True if activations exist, False otherwise.
        If load=True:  return activation data (paths, tensors, etc.)
                       or None if activations don't exist.
        """
        ...

    @abstractmethod
    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Score predictions against ground truth."""
        ...
