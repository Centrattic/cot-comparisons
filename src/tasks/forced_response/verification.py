"""
Verification module for finding questions with high agreement.

Runs N rollouts on each question to find questions where the model
consistently gives the same answer (>80% agreement).

All rollouts run in parallel using asyncio with a configurable number
of workers.
"""

import os
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import openai
from tqdm.asyncio import tqdm as atqdm

from .data_loader import GPQAQuestion, BinaryJudgeQuestion, Question, load_gpqa_questions
from .judge import extract_outcome_from_response
from .prompts import get_question_prompt
from .task import ForcedResponseTask


@dataclass
class RolloutResult:
    """Result of a single rollout."""
    rollout_idx: int
    question_id: str
    prompt: str
    thinking: str
    answer: str
    full_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_idx": self.rollout_idx,
            "prompt": self.prompt,
            "thinking": self.thinking,
            "answer": self.answer,
            "full_response": self.full_response,
        }


def extract_answer(response_text: str) -> str:
    """
    Extract the answer letter from a response.

    Looks for A, B, C, or D in the response, preferring:
    1. The last standalone letter on its own line
    2. The last letter mentioned
    """
    text = response_text.strip()

    # First, look for a standalone letter on its own line (most reliable)
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.upper() in ['A', 'B', 'C', 'D']:
            return line.upper()
        match = re.search(r'(?:answer|choice)(?:\s*(?:is|:))?\s*([A-Da-d])\b', line, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fall back to finding the last mentioned letter
    matches = re.findall(r'\b([A-Da-d])\b', text)
    valid_matches = [m.upper() for m in matches if m.upper() in ['A', 'B', 'C', 'D']]
    if valid_matches:
        return valid_matches[-1]

    return ""


async def run_single_rollout_async(
    async_client: openai.AsyncOpenAI,
    question: Question,
    rollout_idx: int,
    semaphore: asyncio.Semaphore,
    model: str = "moonshotai/kimi-k2-thinking",
    temperature: float = 0.7,
    max_tokens: int = 8000,
) -> RolloutResult:
    """Run a single rollout for a question (async)."""
    # Build prompt based on question type
    if isinstance(question, BinaryJudgeQuestion):
        prompt = question.question  # Binary judge: just the question text
    else:
        prompt = get_question_prompt(question.question, question.choices)

    async with semaphore:
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            message = choice.message

            thinking = ""
            if hasattr(message, 'reasoning') and message.reasoning:
                thinking = message.reasoning
            elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                thinking = message.reasoning_content

            full_response = message.content or ""

            # Extract answer based on question type
            if isinstance(question, BinaryJudgeQuestion):
                answer, is_valid = extract_outcome_from_response(
                    response=full_response,
                    question_type="binary_judge",
                    judge_prompt=question.judge_prompt,
                    bad_outcome=question.bad_outcome,
                )
                if not is_valid:
                    answer = ""
            else:
                answer = extract_answer(full_response)

            return RolloutResult(
                rollout_idx=rollout_idx,
                question_id=question.id,
                prompt=prompt,
                thinking=thinking,
                answer=answer,
                full_response=full_response,
            )

        except Exception as e:
            return RolloutResult(
                rollout_idx=rollout_idx,
                question_id=question.id,
                prompt=prompt,
                thinking="",
                answer="",
                full_response=f"ERROR: {e}",
            )


def compute_summary(question: Question, rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute agreement summary from rollout results."""
    answers = [r["answer"].upper() for r in rollouts if r["answer"]]

    # Valid answers depend on question type
    if isinstance(question, BinaryJudgeQuestion):
        # Binary judge: YES/NO are valid answers
        valid_answers = [a for a in answers if a in ["YES", "NO"]]
    else:
        # Multiple choice: A/B/C/D are valid answers
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

    answer_counts = {}
    for a in valid_answers:
        answer_counts[a] = answer_counts.get(a, 0) + 1

    total_valid = len(valid_answers)
    most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)
    agreement_rate = most_common[1] / total_valid if total_valid > 0 else 0

    summary = {
        "question_id": question.id,
        "question_type": question.question_type,
        "total_rollouts": len(rollouts),
        "valid_rollouts": total_valid,
        "answer_counts": answer_counts,
        "most_common_answer": most_common[0],
        "agreement_rate": agreement_rate,
        "meets_threshold": agreement_rate >= 0.8,
    }

    # Add type-specific fields
    if isinstance(question, BinaryJudgeQuestion):
        # For binary_judge, "correct" means the model did NOT exhibit the bad behavior
        summary["bad_outcome"] = question.bad_outcome
        summary["bad_outcome_rate"] = answer_counts.get(question.bad_outcome.upper(), 0) / total_valid if total_valid > 0 else 0
    else:
        summary["correct_answer"] = question.correct_answer
        summary["is_correct"] = most_common[0] == question.correct_answer

    return summary


async def run_verification_async(
    question: Question,
    num_rollouts: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_workers: int = 250,
    save_results: bool = True,
    verbose: bool = True,
    _async_client: Optional[openai.AsyncOpenAI] = None,
    _semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """
    Run verification rollouts for a single question (async, parallel).

    Args:
        question: The GPQA question to verify
        num_rollouts: Number of rollouts to run
        model: Model to use
        api_key: OpenRouter API key (defaults to env var)
        temperature: Sampling temperature
        max_workers: Maximum concurrent API calls
        save_results: Whether to save results to disk
        verbose: Whether to print progress
        _async_client: Shared async client (for multi-question runs)
        _semaphore: Shared semaphore (for multi-question runs)

    Returns:
        Summary dictionary with agreement statistics
    """
    task = ForcedResponseTask(model=model)

    # Skip if already verified (unless called internally with shared client)
    if _async_client is None:
        existing = task.load_verification_summary(question.id)
        if existing:
            if verbose:
                print(f"  {question.id}: already verified "
                      f"(agreement: {existing['agreement_rate']:.0%}), skipping")
            return existing

    async_client = _async_client or openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )
    semaphore = _semaphore or asyncio.Semaphore(max_workers)

    if verbose:
        print(f"  Launching {num_rollouts} rollouts for {question.id}...")

    # Launch all rollouts concurrently
    jobs = [
        run_single_rollout_async(
            async_client=async_client,
            question=question,
            rollout_idx=i,
            semaphore=semaphore,
            model=model,
            temperature=temperature,
        )
        for i in range(num_rollouts)
    ]

    results = await atqdm.gather(
        *jobs,
        desc=f"  {question.id}",
        total=num_rollouts,
    )

    rollouts = [r.to_dict() for r in results]

    # Save results
    if save_results:
        task.save_verification_result(question, rollouts)

    # Compute summary
    summary = compute_summary(question, rollouts)

    if verbose:
        print(f"\n  Summary for {question.id}:")
        print(f"    Answer distribution: {summary['answer_counts']}")
        print(f"    Agreement rate: {summary['agreement_rate']:.1%}")
        if isinstance(question, BinaryJudgeQuestion):
            print(f"    Bad outcome: {question.bad_outcome}")
            print(f"    Bad outcome rate: {summary['bad_outcome_rate']:.1%}")
        else:
            print(f"    Correct answer: {question.correct_answer}")
            print(f"    Most common: {summary['most_common_answer']} "
                  f"({'correct' if summary['is_correct'] else 'incorrect'})")

    return summary


def run_verification(
    question: Question,
    num_rollouts: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_workers: int = 250,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run verification rollouts (sync wrapper)."""
    return asyncio.run(run_verification_async(
        question=question,
        num_rollouts=num_rollouts,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_workers=max_workers,
        save_results=save_results,
        verbose=verbose,
    ))


async def find_questions_async(
    questions: List[Question],
    num_rollouts: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    max_workers: int = 250,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run verification on all questions in parallel, return all summaries.

    Skips questions that already have verification data on disk.
    Launches remaining questions' rollouts concurrently, limited by max_workers.
    """
    task = ForcedResponseTask(model=model)

    # Check which questions already have verification results
    summaries: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    questions_to_run: List[tuple[int, GPQAQuestion]] = []

    for i, q in enumerate(questions):
        existing = task.load_verification_summary(q.id)
        if existing:
            summaries[i] = existing
            if verbose:
                print(f"  {q.id}: already verified "
                      f"(agreement: {existing['agreement_rate']:.0%}), skipping")
        else:
            questions_to_run.append((i, q))

    if not questions_to_run:
        if verbose:
            print("All questions already verified.")
        return summaries

    async_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )
    semaphore = asyncio.Semaphore(max_workers)

    total_jobs = len(questions_to_run) * num_rollouts
    if verbose:
        print(f"\nLaunching {total_jobs} rollouts across {len(questions_to_run)} questions "
              f"(max {max_workers} concurrent)")
        print()

    # Launch remaining questions concurrently, sharing the semaphore
    tasks = [
        run_verification_async(
            question=q,
            num_rollouts=num_rollouts,
            model=model,
            max_workers=max_workers,
            save_results=True,
            verbose=verbose,
            _async_client=async_client,
            _semaphore=semaphore,
        )
        for _, q in questions_to_run
    ]

    new_summaries = await asyncio.gather(*tasks)

    # Merge results back
    for (idx, _), summary in zip(questions_to_run, new_summaries):
        summaries[idx] = summary

    return summaries


def find_high_agreement_question(
    questions: Optional[List[GPQAQuestion]] = None,
    num_rollouts: int = 50,
    threshold: float = 0.8,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    max_workers: int = 250,
    verbose: bool = True,
) -> Optional[GPQAQuestion]:
    """
    Find a question with high agreement rate.

    Runs all questions in parallel and returns the first one meeting the threshold.
    """
    if questions is None:
        questions = load_gpqa_questions(use_samples=True)

    summaries = asyncio.run(find_questions_async(
        questions=questions,
        num_rollouts=num_rollouts,
        model=model,
        api_key=api_key,
        max_workers=max_workers,
        verbose=verbose,
    ))

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

    for question, summary in zip(questions, summaries):
        if verbose:
            status = "PASS" if summary["agreement_rate"] >= threshold else "FAIL"
            print(f"  {question.id}: {summary['agreement_rate']:.0%} [{status}]")

        if summary["agreement_rate"] >= threshold:
            if verbose:
                print(f"\n*** High-agreement question: {question.id} ***")
            return question

    if verbose:
        print("\nNo question met the threshold.")
    return None


def find_low_agreement_question(
    questions: Optional[List[GPQAQuestion]] = None,
    num_rollouts: int = 50,
    max_agreement: float = 0.5,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    max_workers: int = 250,
    verbose: bool = True,
) -> Optional[GPQAQuestion]:
    """
    Find a question with low agreement rate (spread-out answer distribution).

    Runs all questions in parallel and returns the first one where no answer
    exceeds max_agreement.
    """
    if questions is None:
        questions = load_gpqa_questions(use_samples=True)

    summaries = asyncio.run(find_questions_async(
        questions=questions,
        num_rollouts=num_rollouts,
        model=model,
        api_key=api_key,
        max_workers=max_workers,
        verbose=verbose,
    ))

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

    for question, summary in zip(questions, summaries):
        if verbose:
            status = "PASS" if summary["agreement_rate"] <= max_agreement else "FAIL"
            print(f"  {question.id}: {summary['agreement_rate']:.0%} "
                  f"{summary['answer_counts']} [{status}]")

        if summary["agreement_rate"] <= max_agreement:
            if verbose:
                print(f"\n*** Low-agreement question: {question.id} ***")
            return question

    if verbose:
        print(f"\nNo question found with agreement <= {max_agreement:.0%}.")
    return None
