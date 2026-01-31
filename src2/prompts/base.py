"""
Base prompt ABC for src2.

Prompts bridge tasks and methods: they format task data into LLM prompts
and parse LLM responses back into structured data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePrompt(ABC):
    """
    Abstract base for all prompt objects.

    Subclasses must implement:
      - format(row)           : turn a data row into an LLM prompt string
      - parse_response(text)  : extract structured output from LLM response
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def format(self, row: Dict[str, Any]) -> str:
        """
        Format a single data row into an LLM prompt.

        Args:
            row: dict-like with task-specific keys

        Returns:
            Prompt string ready to send to an LLM.
        """
        ...

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """
        Parse an LLM response into structured data.

        Args:
            response: Raw text from the LLM.

        Returns:
            Parsed result (type depends on the prompt).
        """
        ...
