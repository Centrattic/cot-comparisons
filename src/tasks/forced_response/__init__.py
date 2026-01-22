"""
Forced Response Task

This task analyzes chain-of-thought reasoning by forcing the model to give
a final answer at different points in its reasoning process.

Three modes:
1. Verification: Run N rollouts to find questions with high agreement
2. Force mode: After each sentence of CoT, force the model to give a final answer
3. Resample mode: Force a CoT prefix and run N resamples to see output distribution
"""

from .task import ForcedResponseTask
from .prompts import get_question_prompt, get_force_prompt
from .verification import run_verification
from .forcing import run_forcing
from .resampling import run_resampling, run_resampling_from_verification

__all__ = [
    "ForcedResponseTask",
    "get_question_prompt",
    "get_force_prompt",
    "run_verification",
    "run_forcing",
    "run_resampling",
    "run_resampling_from_verification",
]
