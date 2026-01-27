"""
Prompt templates for the Forced Response task.

Uses GPQA multiple choice questions with Kimi K2 thinking model.
"""

from typing import List, Optional


def get_question_prompt(
    question: str,
    choices: List[str],
    choice_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate a prompt for a multiple choice question.

    Args:
        question: The question text
        choices: List of answer choices
        choice_labels: Optional labels (defaults to A, B, C, D, ...)

    Returns:
        Formatted prompt string
    """
    if choice_labels is None:
        choice_labels = [chr(ord('A') + i) for i in range(len(choices))]

    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    return f"""Answer the following multiple choice question. Think through the problem step by step, then provide your final answer.

Question: {question}

{choices_text}

After your reasoning, provide your final answer as just the letter (A, B, C, or D) on a new line."""


def get_force_prompt(
    question: str,
    choices: List[str],
    partial_cot: str,
    choice_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate a prompt that forces the model to answer after partial CoT.

    This prompt includes the original question and a partial chain of thought,
    then asks the model to give its final answer immediately.

    Args:
        question: The question text
        choices: List of answer choices
        partial_cot: The partial chain of thought to include
        choice_labels: Optional labels (defaults to A, B, C, D, ...)

    Returns:
        Formatted prompt with partial CoT and force instruction
    """
    if choice_labels is None:
        choice_labels = [chr(ord('A') + i) for i in range(len(choices))]

    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    # We structure this as a conversation where the model started reasoning
    # and now needs to give its final answer
    return f"""Answer the following multiple choice question.

Question: {question}

{choices_text}

Here is your reasoning so far:
{partial_cot}

Based on your reasoning so far, what is your final answer? Respond with ONLY the letter (A, B, C, or D). Do not include any other text."""


def get_answer_only_prompt(
    question: str,
    choices: List[str],
    choice_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate a prompt that asks for just the answer (no reasoning).

    Used to test if the model can answer correctly without CoT.

    Args:
        question: The question text
        choices: List of answer choices
        choice_labels: Optional labels (defaults to A, B, C, D, ...)

    Returns:
        Formatted prompt asking for just the letter answer
    """
    if choice_labels is None:
        choice_labels = [chr(ord('A') + i) for i in range(len(choices))]

    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    return f"""Answer the following multiple choice question with ONLY the letter of the correct answer. Do not include any explanation or reasoning.

Question: {question}

{choices_text}

Answer:"""


def split_cot_into_sentences(cot_text: str) -> List[str]:
    """
    Split chain of thought text into sentences.

    Handles common sentence-ending patterns including:
    - Standard punctuation (. ! ?)
    - Newlines as sentence boundaries
    - List items

    Args:
        cot_text: The full chain of thought text

    Returns:
        List of sentences
    """
    import re

    # First, normalize newlines and handle list items
    text = cot_text.strip()

    # Split on sentence-ending punctuation followed by space or newline
    # Also split on double newlines (paragraph breaks)
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$|\n\n+', text)

    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def get_cumulative_cot_segments(cot_text: str) -> List[str]:
    """
    Get cumulative CoT segments for forcing at each sentence boundary.

    For a CoT with sentences [S1, S2, S3], returns:
    [S1, S1+S2, S1+S2+S3]

    Args:
        cot_text: The full chain of thought text

    Returns:
        List of cumulative CoT segments
    """
    sentences = split_cot_into_sentences(cot_text)

    cumulative = []
    current = ""
    for sentence in sentences:
        if current:
            current = current + " " + sentence
        else:
            current = sentence
        cumulative.append(current)

    return cumulative


def build_forcing_prompt(
    question: str,
    choices: List[str],
    partial_cot: str,
    choice_labels: Optional[List[str]] = None,
) -> str:
    """
    Build the full prompt used for forcing, including partial CoT.

    This constructs the text that would be sent to a model when forcing it
    to continue from a partial chain of thought. The format mimics what
    would be prefilled in the model's thinking/assistant turn.

    Args:
        question: The question text
        choices: List of answer choices
        partial_cot: The partial chain of thought prefix
        choice_labels: Optional labels (defaults to A, B, C, D, ...)

    Returns:
        Full prompt string including question, choices, and partial CoT
    """
    if choice_labels is None:
        choice_labels = [chr(ord('A') + i) for i in range(len(choices))]

    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    # Format as a user message followed by assistant's partial thinking
    # This is the format that would be used for activation extraction
    prompt = f"""Answer the following multiple choice question. Think through the problem step by step, then provide your final answer.

Question: {question}

{choices_text}

After your reasoning, provide your final answer as just the letter (A, B, C, or D) on a new line.

<think>
{partial_cot}"""

    return prompt
