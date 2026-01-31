"""
Generate blackmail rollouts using Qwen3-32B via OpenRouter.

Runs N rollouts on the blackmail_001 question and classifies each
using the LLM judge. Returns all rollouts with blackmail flags.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import openai
from tqdm.asyncio import tqdm as atqdm

from ..forced_response.data_loader import BinaryJudgeQuestion, load_custom_questions
from ..forced_response.judge import extract_outcome_from_response_async
from .task import PrincipledResamplingTask

DEFAULT_MODEL = "qwen/qwen3-32b"


@dataclass
class RolloutResult:
    """Result of a single rollout."""
    rollout_idx: int
    prompt: str
    thinking: str
    answer: str
    full_response: str
    is_blackmail: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_idx": self.rollout_idx,
            "prompt": self.prompt,
            "thinking": self.thinking,
            "answer": self.answer,
            "full_response": self.full_response,
            "is_blackmail": self.is_blackmail,
        }


def _load_blackmail_question() -> BinaryJudgeQuestion:
    """Load the blackmail_001 question from questions.json."""
    questions = load_custom_questions()
    for q in questions:
        if isinstance(q, BinaryJudgeQuestion) and q.id == "blackmail_001":
            return q
    raise ValueError("blackmail_001 question not found in questions.json")


async def run_single_rollout_async(
    async_client: openai.AsyncOpenAI,
    question: BinaryJudgeQuestion,
    rollout_idx: int,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 16000,
) -> RolloutResult:
    """Run a single rollout for the blackmail question."""
    prompt = question.question

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

            # Judge for blackmail
            answer, is_valid = await extract_outcome_from_response_async(
                response=full_response,
                question_type="binary_judge",
                judge_prompt=question.judge_prompt,
                bad_outcome=question.bad_outcome,
            )
            if not is_valid:
                answer = ""

            is_blackmail = answer.upper() == "YES" if answer else False

            return RolloutResult(
                rollout_idx=rollout_idx,
                prompt=prompt,
                thinking=thinking,
                answer=answer,
                full_response=full_response,
                is_blackmail=is_blackmail,
            )

        except Exception as e:
            return RolloutResult(
                rollout_idx=rollout_idx,
                prompt=prompt,
                thinking="",
                answer="",
                full_response=f"ERROR: {e}",
                is_blackmail=False,
            )


async def generate_blackmail_rollouts(
    num_rollouts: int = 50,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 16000,
    max_workers: int = 50,
    save: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """
    Generate blackmail rollouts with Qwen3-32B.

    Args:
        num_rollouts: Number of rollouts to generate
        model: OpenRouter model identifier
        api_key: OpenRouter API key
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        max_workers: Max concurrent API calls
        save: Whether to save results to disk
        verbose: Whether to print progress

    Returns:
        List of rollout dicts
    """
    question = _load_blackmail_question()
    task = PrincipledResamplingTask(model=model)

    async_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )
    semaphore = asyncio.Semaphore(max_workers)

    if verbose:
        print(f"Generating {num_rollouts} rollouts with {model}")
        print(f"Max concurrent workers: {max_workers}")
        print()

    # Create run dir if saving
    run_dir = None
    if save:
        config = {
            "model": model,
            "question_id": question.id,
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_workers": max_workers,
        }
        run_dir = task.create_rollout_run_dir(config)
        if verbose:
            print(f"Saving to: {run_dir}")

    # Launch all rollouts
    jobs = [
        run_single_rollout_async(
            async_client=async_client,
            question=question,
            rollout_idx=i,
            semaphore=semaphore,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for i in range(num_rollouts)
    ]

    results = await atqdm.gather(*jobs, desc="Rollouts", total=num_rollouts)
    rollouts = [r.to_dict() for r in results]

    # Save results
    if save and run_dir:
        for rollout in rollouts:
            task.save_rollout(rollout["rollout_idx"], rollout, run_dir)
        task.save_rollout_summary(rollouts, run_dir)

    # Print summary
    blackmail_count = sum(1 for r in rollouts if r.get("is_blackmail"))
    valid_count = sum(1 for r in rollouts if r.get("answer"))

    if verbose:
        print(f"\nResults:")
        print(f"  Total rollouts: {len(rollouts)}")
        print(f"  Valid (judged): {valid_count}")
        print(f"  Blackmail (YES): {blackmail_count}")
        print(f"  Blackmail rate: {blackmail_count/valid_count:.1%}" if valid_count else "  No valid rollouts")
        if blackmail_count > 0:
            indices = [r["rollout_idx"] for r in rollouts if r.get("is_blackmail")]
            print(f"  Blackmail rollout indices: {indices}")

    return rollouts
