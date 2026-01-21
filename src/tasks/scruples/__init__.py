"""
Scruples sycophancy task.

This task uses real-life anecdotes from r/AmItheAsshole to probe sycophantic
tendencies in language models. Three intervention variants are supported:

- first_person: Reframes the anecdote as written by the user ("Am I in the wrong?")
- suggest_right: Appends user's opinion that author is NOT in the wrong
- suggest_wrong: Appends user's opinion that author IS in the wrong

Usage:
    from src.tasks.scruples import ScruplesTask, ScruplesMonitor

    # Create task for a specific variant
    task = ScruplesTask(variant="first_person")

    # Or use convenience classes
    task = ScruplesFirstPersonTask()
    task = ScruplesSuggestRightTask()
    task = ScruplesSuggestWrongTask()

    # Create monitors for analysis
    monitor = ScruplesMonitor(variant="first_person", scope="all_messages")
"""

from .data_loader import (
    compute_consensus,
    filter_high_consensus,
    get_anecdote_by_id,
    load_scruples_data,
)
from .monitor import (
    DEFAULT_MONITOR_MODEL,
    OPENROUTER_BASE_URL,
    ScruplesFirstPersonMonitor,
    ScruplesMonitor,
    ScruplesSuggestRightMonitor,
    ScruplesSuggestWrongMonitor,
)
from .prompts import (
    CONTROL_PROMPT,
    FIRST_PERSON_INTERVENTION_PROMPT,
    INTERVENTION_DESCRIPTIONS,
    INTERVENTION_SUGGESTED_ANSWER,
    MONITOR_PROMPT_TEMPLATE,
    SUGGEST_RIGHT_INTERVENTION_PROMPT,
    SUGGEST_WRONG_INTERVENTION_PROMPT,
    get_control_prompt,
    get_intervention_prompt,
    get_monitor_prompt,
    get_sycophantic_answer,
    is_sycophantic,
)
from .task import (
    ScruplesFirstPersonTask,
    ScruplesSuggestRightTask,
    ScruplesSuggestWrongTask,
    ScruplesTask,
)

__all__ = [
    # Task classes
    "ScruplesTask",
    "ScruplesFirstPersonTask",
    "ScruplesSuggestRightTask",
    "ScruplesSuggestWrongTask",
    # Monitor classes
    "ScruplesMonitor",
    "ScruplesFirstPersonMonitor",
    "ScruplesSuggestRightMonitor",
    "ScruplesSuggestWrongMonitor",
    # Monitor config
    "OPENROUTER_BASE_URL",
    "DEFAULT_MONITOR_MODEL",
    # Data loading
    "load_scruples_data",
    "filter_high_consensus",
    "compute_consensus",
    "get_anecdote_by_id",
    # Prompts
    "CONTROL_PROMPT",
    "FIRST_PERSON_INTERVENTION_PROMPT",
    "SUGGEST_RIGHT_INTERVENTION_PROMPT",
    "SUGGEST_WRONG_INTERVENTION_PROMPT",
    "MONITOR_PROMPT_TEMPLATE",
    "INTERVENTION_DESCRIPTIONS",
    "INTERVENTION_SUGGESTED_ANSWER",
    "get_control_prompt",
    "get_intervention_prompt",
    "get_monitor_prompt",
    "get_sycophantic_answer",
    "is_sycophantic",
]
