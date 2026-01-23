"""
Forced Response Task

This task analyzes chain-of-thought reasoning by forcing the model to give
a final answer at different points in its reasoning process.

Four modes:
1. Verification: Run N rollouts to find questions with high agreement
2. Force mode: True prefill forcing via Tinker — partial CoT is prefilled as
   the start of the model's <think> block, model continues thinking and answers
3. Monitor mode: Partial CoT placed in user message, model predicts answer
4. Resample mode: Force a CoT prefix and run N resamples to see output distribution
"""

from .task import ForcedResponseTask
from .prompts import get_question_prompt, get_force_prompt
from .verification import run_verification
from .forcing import run_forcing, run_forcing_from_verification
from .monitor import run_monitor, run_monitor_from_verification
from .resampling import run_resampling, run_resampling_from_verification

__all__ = [
    "ForcedResponseTask",
    "get_question_prompt",
    "get_force_prompt",
    "run_verification",
    "run_forcing",
    "run_forcing_from_verification",
    "run_monitor",
    "run_monitor_from_verification",
    "run_resampling",
    "run_resampling_from_verification",
]
