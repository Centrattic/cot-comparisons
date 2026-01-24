"""
Monitor (Resampling) module for the Forced Response task.

Predicts what answer the majority of independent continuations would arrive at,
given a partial CoT prefix. Runs at evenly-spaced sentence indices matching
the resampling stride.

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


MAX_RETRIES = 10


@dataclass
class MonitorResamplingResult:
    """Result of a single monitor-resampling attempt."""
    sentence_idx: int
    force_idx: int
    partial_cot: str
    prompt: str
    raw_response: str
    answer: str
    is_valid_single_token: bool
    token_count: int
    num_attempts: int

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
    """Check if response is a valid single-token answer (A, B, C, or D)."""
    text = response.strip()
    if not text:
        return False, "", 0

    tokens = text.split()
    token_count = len(tokens) if tokens else 0

    clean_text = text.upper().rstrip('.')
    if clean_text in ['A', 'B', 'C', 'D']:
        is_valid = len(text) <= 3 and token_count <= 1
        return is_valid, clean_text, token_count

    if re.match(r'^[A-Da-d]\.?\s*$', text):
        answer = text.upper().strip().rstrip('.')
        return True, answer, 1

    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return False, match.group(1).upper(), token_count

    return False, "", token_count


def get_monitor_resampling_messages(
    question: str,
    choices: List[str],
    partial_cot: str,
) -> List[Dict[str, Any]]:
    """
    Create messages for the resampling monitor.

    The prompt explains that 20 independent continuations were sampled from
    this prefix, and asks what answer the majority would arrive at.
    """
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    user_prompt = f"""Question: {question}

{choices_text}

The following is a prefix of a model's internal chain of thought. From this exact prefix, 20 independent continuations were sampled — each time the model completed its reasoning independently from this point and produced a final answer.

<reasoning_prefix>
{partial_cot}
</reasoning_prefix>

Based on this reasoning prefix, what answer would the majority of independent continuations arrive at?

Respond with ONLY the letter (A, B, C, or D). No explanation needed."""

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
) -> MonitorResamplingResult:
    """A single monitor job: retries until a valid single-token answer."""
    messages = get_monitor_resampling_messages(
        question=question.question,
        choices=question.choices,
        partial_cot=partial_cot,
    )
    prompt = messages[0]["content"]

    raw_response = ""
    answer = ""
    token_count = 0

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
                    return MonitorResamplingResult(
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
                if attempt == max_retries - 1:
                    return MonitorResamplingResult(
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

    return MonitorResamplingResult(
        sentence_idx=sentence_idx,
        force_idx=force_idx,
        partial_cot=partial_cot,
        prompt=prompt,
        raw_response=raw_response,
        answer=answer,
        is_valid_single_token=False,
        token_count=token_count,
        num_attempts=max_retries,
    )


async def run_monitor_resampling_async(
    question: GPQAQuestion,
    source_cot: str,
    num_forces: int = 5,
    num_prefix_points: int = 20,
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
    Run resampling monitor at evenly-spaced sentence indices.

    Uses the same stride as actual resampling (~num_prefix_points evenly-spaced).
    """
    async_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )

    task = ForcedResponseTask(model=model)
    semaphore = asyncio.Semaphore(max_workers)

    cot_segments = get_cumulative_cot_segments(source_cot)
    num_sentences = len(cot_segments)

    # Use same stride as resampling
    stride = max(num_sentences // num_prefix_points, 1)
    selected_indices = list(range(0, num_sentences, stride))
    total_jobs = len(selected_indices) * num_forces

    if verbose:
        print(f"Found {num_sentences} sentences in CoT")
        print(f"Correct answer: {question.correct_answer}")
        print(f"Stride: {stride}, prefix points: {len(selected_indices)}")
        print(f"Launching {total_jobs} monitor-resampling jobs with {max_workers} workers")
        print()

    # Create timestamped run directory
    run_dir = None
    if save_results:
        config = {
            "run_type": "monitor_resampling",
            "monitor_model": model,
            "question_id": question.id,
            "rollout_idx": rollout_idx,
            "num_forces": num_forces,
            "num_prefix_points": num_prefix_points,
            "stride": stride,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_sentences": num_sentences,
        }
        run_dir = task.create_run_dir("monitor_resampling", question.id, rollout_idx, config)
        if verbose:
            print(f"Run dir: {run_dir}")
            print()

    # Create all monitor jobs
    jobs = []
    for sentence_idx in selected_indices:
        partial_cot = cot_segments[sentence_idx]
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
        desc="Monitor (resampling)",
        total=total_jobs,
    )

    valid_count = sum(1 for r in results if r.is_valid_single_token)
    if verbose:
        print(f"\nDone. {valid_count}/{total_jobs} valid single-token responses.")
        print()

    # Group results by sentence
    results_by_sentence: Dict[int, List[MonitorResamplingResult]] = {}
    for r in results:
        if r.sentence_idx not in results_by_sentence:
            results_by_sentence[r.sentence_idx] = []
        results_by_sentence[r.sentence_idx].append(r)

    # Process and save results per sentence
    all_sentence_summaries = []
    for sentence_idx in selected_indices:
        sentence_results = results_by_sentence.get(sentence_idx, [])
        partial_cot = cot_segments[sentence_idx]

        if save_results and run_dir:
            task.save_monitor_result(
                question=question,
                sentence_idx=sentence_idx,
                partial_cot=partial_cot,
                force_results=[r.to_dict() for r in sentence_results],
                run_dir=run_dir,
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

    # Save overall summary
    if save_results and run_dir:
        task.save_monitor_summary(
            question=question,
            source_rollout_idx=rollout_idx,
            all_sentence_results=all_sentence_summaries,
            source_cot=source_cot,
            run_dir=run_dir,
        )

    return {
        "question_id": question.id,
        "correct_answer": question.correct_answer,
        "num_sentences": num_sentences,
        "num_prefix_points": len(selected_indices),
        "stride": stride,
        "sentence_results": all_sentence_summaries,
    }


def run_monitor_resampling(
    question: GPQAQuestion,
    source_cot: str,
    num_forces: int = 5,
    num_prefix_points: int = 20,
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
    """Run resampling monitor (sync wrapper)."""
    return asyncio.run(run_monitor_resampling_async(
        question=question,
        source_cot=source_cot,
        num_forces=num_forces,
        num_prefix_points=num_prefix_points,
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


def run_monitor_resampling_from_verification(
    question_id: str,
    rollout_idx: int = 0,
    num_forces: int = 5,
    num_prefix_points: int = 20,
    max_workers: int = 300,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run resampling monitor using a CoT from verification rollouts."""
    task = ForcedResponseTask(model=model)

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

    source_cot = rollout_data.get("thinking", "")
    if not source_cot:
        source_cot = rollout_data.get("full_response", "")

    if not source_cot:
        print(f"No CoT found in rollout {rollout_idx}")
        return None

    if verbose:
        print(f"Running monitor-resampling for {question_id}")
        print(f"Using rollout {rollout_idx}")
        print(f"CoT length: {len(source_cot)} characters")
        print()

    return run_monitor_resampling(
        question=question,
        source_cot=source_cot,
        num_forces=num_forces,
        num_prefix_points=num_prefix_points,
        max_workers=max_workers,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        rollout_idx=rollout_idx,
    )
