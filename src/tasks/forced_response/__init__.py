"""
Forced Response Task

This task analyzes chain-of-thought reasoning by forcing the model to give
a final answer at different points in its reasoning process.

Two modes:
1. Force mode: After each sentence of CoT, force the model to give a final answer
2. Resample mode: From a given point, resample the model's answer distribution (future)
"""

from .task import ForcedResponseTask
from .prompts import get_question_prompt, get_force_prompt
from .verification import run_verification
from .forcing import run_forcing

__all__ = [
    "ForcedResponseTask",
    "get_question_prompt",
    "get_force_prompt",
    "run_verification",
    "run_forcing",
]
