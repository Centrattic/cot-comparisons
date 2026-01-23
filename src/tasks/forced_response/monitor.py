"""
Monitor module for the Forced Response task.

After every sentence of the chain of thought, we put the partial CoT in the
user message and ask the model to predict an answer. This is a "monitor" approach
(not true forcing) since the model sees the CoT as context rather than having
it prefilled as its own reasoning.

All monitor jobs run in parallel using asyncio with a configurable number
of workers.
"""

import os
import json
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import openai
from tqdm.asyncio import tqdm as atqdm

from .data_loader import GPQAQuestion
from .prompts import get_cumulative_cot_segments
from .task import ForcedResponseTask


MAX_RETRIES = 10  # Max API calls per monitor job before giving up


@dataclass
class MonitorResult:
    """Result of a single monitor attempt."""
    sentence_idx: int
    force_idx: int
    partial_cot: str
    prompt: str
    raw_response: str
    answer: str
    is_valid_single_token: bool
    token_count: int
    num_attempts: int  # How many API calls it took

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx,
            "force_idx": self.force_idx,
            "partial_cot": self.partial_cot,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "answer": self.answer,
            "is_valid_single_token": self.is_valid_single_token,
            "token_count": self.token_count,
            "num_attempts": self.num_attempts,
        }


def is_single_token_answer(response: str) -> Tuple[bool, str, int]:
    """
    Check if response is a valid single-token answer.

    A valid single-token answer is just the letter A, B, C, or D,
    possibly with whitespace or minimal punctuation.

    Args:
        response: The model's response text

    Returns:
        Tuple of (is_valid, extracted_answer, approximate_token_count)
    """
    text = response.strip()

    if not text:
        return False, "", 0

    tokens = text.split()
    token_count = len(tokens) if tokens else 0

    # Valid patterns: "A", "B", "C", "D", "A.", "B.", etc.
    clean_text = text.upper().rstrip('.')

    if clean_text in ['A', 'B', 'C', 'D']:
        is_valid = len(text) <= 3 and token_count <= 1
        return is_valid, clean_text, token_count

    # Also accept if the response is just the letter with some whitespace
    if re.match(r'^[A-Da-d]\.?\s*$', text):
        answer = text.upper().strip().rstrip('.')
        return True, answer, 1

    # Not a single token answer - try to extract the answer anyway
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return False, match.group(1).upper(), token_count

    return False, "", token_count


def get_monitor_messages(
    question: str,
    choices: List[str],
    partial_cot: str,
) -> List[Dict[str, Any]]:
    """
    Create messages for monitoring the model's answer based on partial reasoning.
    """
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    user_prompt = f"""Question: {question}

{choices_text}

Here is reasoning about this problem so far:
<reasoning>
{partial_cot}
</reasoning>

Based on the reasoning above, what is the answer? Respond with ONLY the letter (A, B, C, or D). No explanation needed."""

    return [{"role": "user", "content": user_prompt}]


async def run_single_monitor_job(
    async_client: openai.AsyncOpenAI,
    question: GPQAQuestion,
    partial_cot: str,
    sentence_idx: int,
    force_idx: int,
    semaphore: asyncio.Semaphore,
    model: str = "moonshotai/kimi-k2-thinking",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = MAX_RETRIES,
) -> MonitorResult:
    """
    A single monitor job: retries API calls until a valid single-token answer.

    Args:
        async_client: Async OpenAI client
        question: The GPQA question
        partial_cot: The partial chain of thought
        sentence_idx: Index of the sentence
        force_idx: Index of this monitor attempt
        semaphore: Semaphore for concurrency control
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        max_retries: Max API calls before giving up

    Returns:
        MonitorResult with the valid response (or best effort after max_retries)
    """
    messages = get_monitor_messages(
        question=question.question,
        choices=question.choices,
        partial_cot=partial_cot,
    )
    prompt = messages[0]["content"]

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                raw_response = response.choices[0].message.content or ""
                is_valid, answer, token_count = is_single_token_answer(raw_response)

                if is_valid:
                    return MonitorResult(
                        sentence_idx=sentence_idx,
                        force_idx=force_idx,
                        partial_cot=partial_cot,
                        prompt=prompt,
                        raw_response=raw_response,
                        answer=answer,
                        is_valid_single_token=True,
                        token_count=token_count,
                        num_attempts=attempt + 1,
                    )

            except Exception as e:
                # On error, continue retrying
                if attempt == max_retries - 1:
                    return MonitorResult(
                        sentence_idx=sentence_idx,
                        force_idx=force_idx,
                        partial_cot=partial_cot,
                        prompt=prompt,
                        raw_response=f"ERROR: {e}",
                        answer="",
                        is_valid_single_token=False,
                        token_count=0,
                        num_attempts=attempt + 1,
                    )

    # Exhausted retries without a valid single-token answer
    return MonitorResult(
        sentence_idx=sentence_idx,
        force_idx=force_idx,
        partial_cot=partial_cot,
        prompt=prompt,
        raw_response=raw_response,  # Last response
        answer=answer,  # Best-effort extracted answer
        is_valid_single_token=False,
        token_count=token_count,
        num_attempts=max_retries,
    )


async def run_monitor_async(
    question: GPQAQuestion,
    source_cot: str,
    num_forces: int = 5,
    max_workers: int = 300,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = MAX_RETRIES,
    save_results: bool = True,
    verbose: bool = True,
    rollout_idx: int = 0,
) -> Dict[str, Any]:
    """
    Run monitoring for all sentences in parallel.

    All monitor jobs (num_forces * num_sentences) are launched concurrently,
    limited by the semaphore to max_workers parallel API calls.

    Args:
        question: The GPQA question
        source_cot: The full chain of thought
        num_forces: Number of monitor attempts per sentence
        max_workers: Maximum concurrent API calls
        model: Model to use
        api_key: OpenRouter API key
        temperature: Sampling temperature
        max_tokens: Maximum tokens per response
        max_retries: Max retries per monitor job
        save_results: Whether to save results to disk
        verbose: Whether to print progress
        rollout_idx: Index of the source rollout being monitored

    Returns:
        Summary dictionary with all monitoring results
    """
    async_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )

    task = ForcedResponseTask(model=model)
    semaphore = asyncio.Semaphore(max_workers)

    # Get cumulative CoT segments (after each sentence)
    cot_segments = get_cumulative_cot_segments(source_cot)
    num_sentences = len(cot_segments)
    total_jobs = num_sentences * num_forces

    if verbose:
        print(f"Found {num_sentences} sentences in CoT")
        print(f"Correct answer: {question.correct_answer}")
        print(f"Launching {total_jobs} monitor jobs with {max_workers} workers")
        print()

    # Create all monitor jobs
    jobs = []
    for sentence_idx, partial_cot in enumerate(cot_segments):
        for force_idx in range(num_forces):
            job = run_single_monitor_job(
                async_client=async_client,
                question=question,
                partial_cot=partial_cot,
                sentence_idx=sentence_idx,
                force_idx=force_idx,
                semaphore=semaphore,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            jobs.append(job)

    # Run all jobs concurrently with progress bar
    results = await atqdm.gather(
        *jobs,
        desc="Monitoring",
        total=total_jobs,
    )

    valid_count = sum(1 for r in results if r.is_valid_single_token)
    if verbose:
        print(f"\nDone. {valid_count}/{total_jobs} valid single-token responses.")
        print()

    # Group results by sentence
    results_by_sentence: Dict[int, List[MonitorResult]] = {}
    for r in results:
        if r.sentence_idx not in results_by_sentence:
            results_by_sentence[r.sentence_idx] = []
        results_by_sentence[r.sentence_idx].append(r)

    # Process and save results per sentence
    all_sentence_summaries = []
    for sentence_idx in range(num_sentences):
        sentence_results = results_by_sentence.get(sentence_idx, [])
        partial_cot = cot_segments[sentence_idx]

        if save_results:
            task.save_monitor_result(
                question=question,
                sentence_idx=sentence_idx,
                partial_cot=partial_cot,
                force_results=[r.to_dict() for r in sentence_results],
                rollout_idx=rollout_idx,
            )

        answers = [r.answer for r in sentence_results if r.answer]
        valid_results = [r for r in sentence_results if r.is_valid_single_token]

        answer_counts: Dict[str, int] = {}
        for a in answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        sentence_summary = {
            "sentence_idx": sentence_idx,
            "partial_cot_length": len(partial_cot),
            "total_forces": len(sentence_results),
            "valid_single_token": len(valid_results),
            "answer_counts": answer_counts,
            "most_common": max(answer_counts.items(), key=lambda x: x[1])[0] if answer_counts else "",
        }
        all_sentence_summaries.append(sentence_summary)

        if verbose:
            print(f"  Sentence {sentence_idx + 1}/{num_sentences}: "
                  f"{answer_counts}, valid: {len(valid_results)}/{len(sentence_results)}")

    # Save overall summary
    if save_results:
        task.save_monitor_summary(
            question=question,
            source_rollout_idx=rollout_idx,
            all_sentence_results=all_sentence_summaries,
            source_cot=source_cot,
        )

    return {
        "question_id": question.id,
        "correct_answer": question.correct_answer,
        "num_sentences": num_sentences,
        "sentence_results": all_sentence_summaries,
    }


def run_monitor(
    question: GPQAQuestion,
    source_cot: str,
    num_forces: int = 5,
    max_workers: int = 300,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = MAX_RETRIES,
    save_results: bool = True,
    verbose: bool = True,
    rollout_idx: int = 0,
) -> Dict[str, Any]:
    """
    Run monitoring for all sentences (sync wrapper around async implementation).
    """
    return asyncio.run(run_monitor_async(
        question=question,
        source_cot=source_cot,
        num_forces=num_forces,
        max_workers=max_workers,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        save_results=save_results,
        verbose=verbose,
        rollout_idx=rollout_idx,
    ))


def run_monitor_from_verification(
    question_id: str,
    rollout_idx: int = 0,
    num_forces: int = 5,
    max_workers: int = 300,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run monitoring using a CoT from verification rollouts.
    """
    task = ForcedResponseTask(model=model)

    # Load verification data
    verification_dir = task.verification_dir / question_id
    if not verification_dir.exists():
        print(f"No verification data found for {question_id}")
        return None

    rollout_path = verification_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
    if not rollout_path.exists():
        print(f"Rollout {rollout_idx} not found for {question_id}")
        return None

    with open(rollout_path) as f:
        rollout_data = json.load(f)

    summary_path = verification_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    question = GPQAQuestion(
        id=summary["question_id"],
        question=summary["question"],
        choices=summary["choices"],
        correct_answer=summary["correct_answer"],
        correct_index=ord(summary["correct_answer"]) - ord('A'),
    )

    # Get the CoT from the rollout
    source_cot = rollout_data.get("thinking", "")
    if not source_cot:
        source_cot = rollout_data.get("full_response", "")

    if not source_cot:
        print(f"No CoT found in rollout {rollout_idx}")
        return None

    if verbose:
        print(f"Running monitor for {question_id}")
        print(f"Using rollout {rollout_idx}")
        print(f"CoT length: {len(source_cot)} characters")
        print()

    return run_monitor(
        question=question,
        source_cot=source_cot,
        num_forces=num_forces,
        max_workers=max_workers,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        rollout_idx=rollout_idx,
    )
