"""
Core resampling module for Thought Branches CI++ computation.

For each sentence Si in a CoT, truncates before Si and generates N
continuations via OpenRouter. Each continuation is judged for blackmail.

Supports two prefill strategies:
1. Assistant prefill with continue_final_message (preferred)
2. Prompt-based continuation (fallback)
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import openai
from tqdm.asyncio import tqdm as atqdm

from ..forced_response.data_loader import BinaryJudgeQuestion, load_custom_questions
from ..forced_response.judge import judge_response_async
from ..forced_response.prompts import split_cot_into_sentences

DEFAULT_MODEL = "qwen/qwen3-32b"


@dataclass
class ResampleResult:
    """Result of a single resample continuation."""
    sentence_idx: int
    resample_idx: int
    prefix: str
    continuation_thinking: str
    full_response: str
    answer: str
    is_blackmail: bool
    continuation_sentences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx,
            "resample_idx": self.resample_idx,
            "prefix": self.prefix,
            "continuation_thinking": self.continuation_thinking,
            "full_response": self.full_response,
            "answer": self.answer,
            "is_blackmail": self.is_blackmail,
            "continuation_sentences": self.continuation_sentences,
        }


def _load_blackmail_question() -> BinaryJudgeQuestion:
    """Load the blackmail_001 question from questions.json."""
    questions = load_custom_questions()
    for q in questions:
        if isinstance(q, BinaryJudgeQuestion) and q.id == "blackmail_001":
            return q
    raise ValueError("blackmail_001 question not found in questions.json")


async def resample_from_prefix_async(
    async_client: openai.AsyncOpenAI,
    question: BinaryJudgeQuestion,
    prefix: str,
    resample_idx: int,
    sentence_idx: int,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 16000,
) -> ResampleResult:
    """
    Generate one resample continuation from a CoT prefix.

    Uses assistant prefill: passes the partial CoT as the start of the
    assistant's thinking, and asks the model to continue.
    """
    async with semaphore:
        try:
            # Build messages with assistant prefill
            messages = [
                {"role": "user", "content": question.question},
            ]

            if prefix.strip():
                # Add partial assistant message to continue from
                messages.append({
                    "role": "assistant",
                    "content": f"<think>\n{prefix}",
                    "prefix": True,
                })

            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            message = choice.message

            # Extract thinking (continuation after our prefix)
            thinking = ""
            if hasattr(message, 'reasoning') and message.reasoning:
                thinking = message.reasoning
            elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                thinking = message.reasoning_content

            full_response = message.content or ""

            # The continuation is the new thinking generated after the prefix
            continuation_thinking = thinking

            # Judge for blackmail
            is_blackmail_result, raw_judge = await judge_response_async(
                response=full_response,
                judge_prompt=question.judge_prompt,
                bad_outcome=question.bad_outcome,
            )
            answer = "YES" if is_blackmail_result else "NO"

            # Split continuation into sentences for similarity analysis
            continuation_sentences = split_cot_into_sentences(continuation_thinking) if continuation_thinking else []

            return ResampleResult(
                sentence_idx=sentence_idx,
                resample_idx=resample_idx,
                prefix=prefix,
                continuation_thinking=continuation_thinking,
                full_response=full_response,
                answer=answer,
                is_blackmail=is_blackmail_result,
                continuation_sentences=continuation_sentences,
            )

        except Exception as e:
            return ResampleResult(
                sentence_idx=sentence_idx,
                resample_idx=resample_idx,
                prefix=prefix,
                continuation_thinking="",
                full_response=f"ERROR: {e}",
                answer="",
                is_blackmail=False,
                continuation_sentences=[],
            )


async def resample_sentence(
    question: BinaryJudgeQuestion,
    source_cot_sentences: List[str],
    sentence_idx: int,
    num_resamples: int = 100,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 16000,
    max_workers: int = 50,
    verbose: bool = True,
) -> List[ResampleResult]:
    """
    Resample continuations for a single sentence position.

    Truncates the CoT before sentence_idx and generates num_resamples
    continuations from that prefix.
    """
    # Build prefix: everything before sentence_idx
    prefix = " ".join(source_cot_sentences[:sentence_idx])

    async_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )
    semaphore = asyncio.Semaphore(max_workers)

    jobs = [
        resample_from_prefix_async(
            async_client=async_client,
            question=question,
            prefix=prefix,
            resample_idx=j,
            sentence_idx=sentence_idx,
            semaphore=semaphore,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for j in range(num_resamples)
    ]

    desc = f"  Sentence {sentence_idx}"
    results = await atqdm.gather(*jobs, desc=desc, total=num_resamples)

    return list(results)


async def resample_all_sentences(
    source_cot: str,
    num_resamples: int = 100,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 16000,
    max_workers: int = 50,
    verbose: bool = True,
) -> Dict[int, List[ResampleResult]]:
    """
    Resample for every sentence in a CoT.

    Args:
        source_cot: The full CoT text to analyze
        num_resamples: Number of resamples per sentence
        model: OpenRouter model identifier
        api_key: OpenRouter API key
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        max_workers: Max concurrent API calls
        verbose: Whether to print progress

    Returns:
        Dict mapping sentence_idx -> list of ResampleResults
    """
    question = _load_blackmail_question()
    sentences = split_cot_into_sentences(source_cot)

    if verbose:
        print(f"CoT has {len(sentences)} sentences")
        print(f"Generating {num_resamples} resamples per sentence")
        print(f"Total API calls: {len(sentences) * num_resamples}")
        print()

    results = {}
    for i in range(len(sentences)):
        if verbose:
            print(f"Resampling sentence {i}/{len(sentences)-1}: "
                  f"{sentences[i][:80]}{'...' if len(sentences[i]) > 80 else ''}")

        sentence_results = await resample_sentence(
            question=question,
            source_cot_sentences=sentences,
            sentence_idx=i,
            num_resamples=num_resamples,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
            verbose=verbose,
        )
        results[i] = sentence_results

        if verbose:
            blackmail_count = sum(1 for r in sentence_results if r.is_blackmail)
            valid_count = sum(1 for r in sentence_results if r.answer)
            rate = blackmail_count / valid_count if valid_count > 0 else 0
            print(f"    -> P(blackmail) = {rate:.2f} ({blackmail_count}/{valid_count})")

    return results
