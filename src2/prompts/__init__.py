from .scruples import (
    ScruplesBaseMonitorPrompt,
    ScruplesDiscriminationPrompt,
    ScruplesBaselinePrompt,
    # Raw templates and helpers (used by ScruplesTask)
    get_control_prompt,
    get_intervention_prompt,
    is_sycophantic,
    INTERVENTION_DESCRIPTIONS,
    MONITOR_PROMPT_TEMPLATE,
    THINKING_BLOCK_TEMPLATE,
    DISCRIMINATION_MONITOR_PROMPT_TEMPLATE,
    BASELINE_PROMPT_TEMPLATE,
)
from .forced_response import (
    ForcingMonitorPrompt,
    ResamplingMonitorPrompt,
    # Raw templates and helpers (used by ForcingTask/ResamplingTask)
    get_question_prompt,
    get_force_prompt,
    get_answer_only_prompt,
    build_forcing_prompt,
    get_cumulative_cot_segments,
    split_cot_into_sentences,
)
