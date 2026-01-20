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
    load_scruples_data,
    prepare_task_data,
    filter_author_wrong,
    get_anecdote_by_id,
)
from .monitor import (
    ScruplesMonitor,
    ScruplesFirstPersonMonitor,
    ScruplesSuggestRightMonitor,
    ScruplesSuggestWrongMonitor,
    OPENROUTER_BASE_URL,
    DEFAULT_MONITOR_MODEL,
)
from .prompts import (
    CONTROL_PROMPT,
    FIRST_PERSON_INTERVENTION_PROMPT,
    SUGGEST_RIGHT_INTERVENTION_PROMPT,
    SUGGEST_WRONG_INTERVENTION_PROMPT,
    MONITOR_PROMPT_TEMPLATE,
    HINT_DESCRIPTIONS,
    SYCOPHANTIC_ANSWER,
    get_control_prompt,
    get_intervention_prompt,
    get_monitor_prompt,
)
from .task import (
    ScruplesTask,
    ScruplesFirstPersonTask,
    ScruplesSuggestRightTask,
    ScruplesSuggestWrongTask,
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
    "prepare_task_data",
    "filter_author_wrong",
    "get_anecdote_by_id",
    # Prompts
    "CONTROL_PROMPT",
    "FIRST_PERSON_INTERVENTION_PROMPT",
    "SUGGEST_RIGHT_INTERVENTION_PROMPT",
    "SUGGEST_WRONG_INTERVENTION_PROMPT",
    "MONITOR_PROMPT_TEMPLATE",
    "HINT_DESCRIPTIONS",
    "SYCOPHANTIC_ANSWER",
    "get_control_prompt",
    "get_intervention_prompt",
    "get_monitor_prompt",
]
