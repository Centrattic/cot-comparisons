"""
LLM judge for classifying binary_judge question responses.

Uses OpenRouter API with a fast model (gpt-4o-mini) to classify
whether a response exhibits the specified bad behavior.
"""

import os
import threading
from typing import Optional, Tuple

import openai


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

# Thread-local storage for sync clients (avoids creating new client per call)
_thread_local = threading.local()


def _get_sync_client(api_key: Optional[str] = None) -> openai.OpenAI:
    """Get or create a thread-local sync OpenAI client."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not hasattr(_thread_local, "client") or _thread_local.client_key != key:
        _thread_local.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        _thread_local.client_key = key
    return _thread_local.client


def judge_response(
    response: str,
    judge_prompt: str,
    bad_outcome: str,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Tuple[bool, str]:
    """
    Use an LLM to judge whether a response exhibits a specific behavior.

    Args:
        response: The model's full response to evaluate.
        judge_prompt: Prompt template with {response} placeholder.
        bad_outcome: What the judge returns if model misbehaved (e.g., "YES").
        model: Judge model to use.
        api_key: OpenRouter API key.
        max_retries: Number of retry attempts.

    Returns:
        Tuple of (is_bad_outcome, raw_judge_response).
    """
    client = _get_sync_client(api_key)
    filled_prompt = judge_prompt.format(response=response)

    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": filled_prompt}],
                max_tokens=10,
                temperature=0,
            )
            raw_response = result.choices[0].message.content or ""
            raw_response = raw_response.strip().upper()

            is_bad = bad_outcome.upper() in raw_response
            return is_bad, raw_response

        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"ERROR: {e}"

    return False, "ERROR: max retries exceeded"


def extract_binary_answer(
    response: str,
    judge_prompt: str,
    bad_outcome: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> str:
    """
    Convenience wrapper returning "YES" or "NO" (or "" on error).

    Used by verification rollout processing for binary_judge questions.
    """
    is_bad, raw = judge_response(
        response, judge_prompt, bad_outcome, model=judge_model,
    )
    if "ERROR" in raw:
        return ""
    return bad_outcome.upper() if is_bad else ("NO" if bad_outcome.upper() == "YES" else "YES")
