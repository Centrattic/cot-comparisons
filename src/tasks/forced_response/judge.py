"""
LLM judge for classifying model responses.

Used for binary_judge question types where we need to determine if the model
exhibited a specific behavior (e.g., blackmail).
"""

import os
import re
import asyncio
from typing import Optional

import openai


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


async def judge_response_async(
    response: str,
    judge_prompt: str,
    bad_outcome: str,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> tuple[bool, str]:
    """
    Use an LLM to judge whether a response exhibits a specific behavior.

    Args:
        response: The model's full response to evaluate
        judge_prompt: Prompt template with {response} placeholder
        bad_outcome: What the judge returns if model misbehaved (e.g., "YES")
        model: Judge model to use (default: gpt-4o-mini)
        api_key: OpenRouter API key
        max_retries: Number of retry attempts

    Returns:
        Tuple of (is_bad_outcome: bool, raw_judge_response: str)
    """
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )

    filled_prompt = judge_prompt.format(response=response)

    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": filled_prompt}],
                max_tokens=10,
                temperature=0,
            )
            raw_response = result.choices[0].message.content or ""
            raw_response = raw_response.strip().upper()

            # Check if response matches bad outcome
            is_bad = bad_outcome.upper() in raw_response

            return is_bad, raw_response

        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"ERROR: {e}"

    return False, "ERROR: max retries exceeded"


def judge_response(
    response: str,
    judge_prompt: str,
    bad_outcome: str,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> tuple[bool, str]:
    """Sync wrapper for judge_response_async."""
    return asyncio.run(judge_response_async(
        response=response,
        judge_prompt=judge_prompt,
        bad_outcome=bad_outcome,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
    ))


def extract_outcome_from_response(
    response: str,
    question_type: str,
    choices: Optional[list] = None,
    judge_prompt: Optional[str] = None,
    bad_outcome: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[str, bool]:
    """
    Extract outcome from a response based on question type.

    For multiple_choice: extracts A/B/C/D
    For binary_judge: calls LLM judge to classify

    Args:
        response: The model's response
        question_type: "multiple_choice" or "binary_judge"
        choices: List of choices (for multiple_choice)
        judge_prompt: Judge prompt template (for binary_judge)
        bad_outcome: What judge returns for bad behavior (for binary_judge)
        judge_model: Model to use for judging

    Returns:
        Tuple of (outcome: str, is_valid: bool)
        - For multiple_choice: ("A", True) or ("", False)
        - For binary_judge: ("YES", True) or ("NO", True) or ("ERROR", False)
    """
    if question_type == "binary_judge":
        if not judge_prompt or not bad_outcome:
            return "ERROR", False

        is_bad, raw = judge_response(
            response=response,
            judge_prompt=judge_prompt,
            bad_outcome=bad_outcome,
            model=judge_model,
        )

        if "ERROR" in raw:
            return raw, False

        return raw, True

    else:
        # Multiple choice - extract A/B/C/D
        # Look for answer after </think> if present
        if "</think>" in response:
            parts = response.split("</think>", 1)
            answer_text = parts[1].strip() if len(parts) > 1 else ""
        else:
            answer_text = response

        # Try exact single letter
        clean = answer_text.upper().rstrip(".").strip()
        if clean in ["A", "B", "C", "D"]:
            return clean, True

        # Try to find letter in text
        match = re.search(r"\b([A-Da-d])\b", answer_text)
        if match:
            return match.group(1).upper(), True

        return "", False
