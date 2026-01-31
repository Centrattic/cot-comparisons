"""
Base method ABC for src2.

Methods are data consumers: they receive data from tasks, optionally train,
and produce predictions. Output is managed via OutputManager.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.output import OutputManager


class BaseMethod(ABC):
    """
    Abstract base for all methods.

    Lifecycle:
      1. m = SomeMethod(...)
      2. m.set_task(task)        -> binds method to a task
      3. folder = m.get_folder() -> creates timestamped output dir (saves config)
      4. m.train(data)  (optional)
      5. m.infer(data)
      6. m._output.mark_success()  -> updates `latest` symlink
    """

    name: str

    def __init__(self, name: str):
        self.name = name
        self._task = None
        self._output: Optional[OutputManager] = None

    def set_task(self, task) -> None:
        """Bind this method to a task. Creates an OutputManager for it."""
        self._task = task
        base_dir = task.data_dir / self.name
        self._output = OutputManager(base_dir)

    def get_folder(self) -> Path:
        """Create and return a timestamped run folder for this method+task.

        Automatically saves method and task config to method_config.json.
        """
        if self._output is None:
            raise RuntimeError("Call set_task() before get_folder().")
        folder = self._output.create_run_folder()
        config = {
            "method": self.get_config(),
        }
        if self._task is not None:
            config["task"] = self._task.get_config()
        with open(folder / "method_config.json", "w") as f:
            json.dump(config, f, indent=2)
        return folder

    def get_config(self) -> Dict[str, Any]:
        """Return a dict of this method's configuration for serialization.

        Default implementation returns all public, JSON-serializable attributes.
        Subclasses can override for custom behavior.
        """
        config = {"name": self.name}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                config[key] = value
            elif hasattr(value, '__dict__') and hasattr(value, '__dataclass_fields__'):
                # Serialize dataclasses (like ProbeConfig)
                from dataclasses import asdict
                config[key] = asdict(value)
        return config

    @abstractmethod
    def infer(self, data: Any) -> Any:
        """Run inference on task data and save results to get_folder()."""
        ...

    def train(self, data: Any) -> None:
        """Optional: train on prepared data. Default is no-op."""
        pass
