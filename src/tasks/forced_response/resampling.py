"""
Resampling module for the Forced Response task.

Unlike forcing (which asks for just the answer given partial CoT as context),
resampling forces the partial CoT as the assistant's prefix and lets the model
continue generating. We run N resamples to see the output distribution.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import openai

from .data_loader import GPQAQuestion
from .prompts import get_question_prompt, get_cumulative_cot_segments
from .verification import extract_answer
from .task import ForcedResponseTask


@dataclass
class ResampleResult:
    """Result of a single resample attempt."""
    sentence_idx: int
    resample_idx: int
    forced_prefix: str
    continuation: str
    full_response: str
    thinking: str
    answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx,
            "resample_idx": self.resample_idx,
            "forced_prefix": self.forced_prefix,
            "continuation": self.continuation,
            "full_response": self.full_response,
            "thinking": self.thinking,
            "answer": self.answer,
        }


def get_resample_messages(
    question: str,
    choices: List[str],
    forced_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Create messages for resampling with a forced prefix.

    Uses assistant prefill: we provide the partial CoT as an incomplete
    assistant message, and the model continues from there.

    Args:
        question: The question text
        choices: List of answer choices
        forced_prefix: The partial CoT to force as prefix

    Returns:
        List of messages with assistant prefill
    """
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    user_prompt = f"""Answer the following multiple choice question. Think through the problem step by step, then provide your final answer.

Question: {question}

{choices_text}

After your reasoning, provide your final answer as just the letter (A, B, C, or D) on a new line."""

    # Use assistant prefill to force the prefix
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": forced_prefix},
    ]
    return messages


def run_single_resample(
    client: openai.OpenAI,
    question: GPQAQuestion,
    forced_prefix: str,
    sentence_idx: int,
    resample_idx: int,
    model: str = "moonshotai/kimi-k2-thinking",
    temperature: float = 0.7,
    max_tokens: int = 8000,
) -> ResampleResult:
    """
    Run a single resample by forcing a prefix and letting the model continue.

    Args:
        client: OpenAI client configured for OpenRouter
        question: The GPQA question
        forced_prefix: The partial CoT to force as prefix
        sentence_idx: Index of the sentence we're resampling at
        resample_idx: Index of this resample attempt
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for continuation

    Returns:
        ResampleResult with the continuation and answer
    """
    messages = get_resample_messages(
        question=question.question,
        choices=question.choices,
        forced_prefix=forced_prefix,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract thinking (for thinking models)
        thinking = ""
        if hasattr(message, 'reasoning') and message.reasoning:
            thinking = message.reasoning
        elif hasattr(message, 'reasoning_content') and message.reasoning_content:
            thinking = message.reasoning_content

        # The continuation is what the model generated
        continuation = message.content or ""

        # Full response is prefix + continuation
        full_response = forced_prefix + continuation

        # Extract answer from the full response
        answer = extract_answer(full_response)

        return ResampleResult(
            sentence_idx=sentence_idx,
            resample_idx=resample_idx,
            forced_prefix=forced_prefix,
            continuation=continuation,
            full_response=full_response,
            thinking=thinking,
            answer=answer,
        )

    except Exception as e:
        print(f"Error in resample {resample_idx} at sentence {sentence_idx}: {e}")
        return ResampleResult(
            sentence_idx=sentence_idx,
            resample_idx=resample_idx,
            forced_prefix=forced_prefix,
            continuation=f"ERROR: {e}",
            full_response=forced_prefix,
            thinking="",
            answer="",
        )


def run_resampling_for_sentence(
    client: openai.OpenAI,
    question: GPQAQuestion,
    forced_prefix: str,
    sentence_idx: int,
    num_resamples: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    temperature: float = 0.7,
    verbose: bool = True,
) -> List[ResampleResult]:
    """
    Run multiple resamples for a single sentence position.

    Args:
        client: OpenAI client configured for OpenRouter
        question: The GPQA question
        forced_prefix: The partial CoT to force as prefix
        sentence_idx: Index of the sentence
        num_resamples: Number of resample attempts
        model: Model to use
        temperature: Sampling temperature
        verbose: Whether to print progress

    Returns:
        List of ResampleResult objects
    """
    results = []

    for i in range(num_resamples):
        if verbose:
            print(f"    Resample {i+1}/{num_resamples}...", end=" ", flush=True)

        result = run_single_resample(
            client=client,
            question=question,
            forced_prefix=forced_prefix,
            sentence_idx=sentence_idx,
            resample_idx=i,
            model=model,
            temperature=temperature,
        )
        results.append(result)

        if verbose:
            print(f"Answer: {result.answer or 'N/A'}")

    return results


def run_resampling(
    question: GPQAQuestion,
    source_cot: str,
    num_resamples: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    sentence_indices: Optional[List[int]] = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run resampling for sentences in a chain of thought.

    Args:
        question: The GPQA question
        source_cot: The full chain of thought to resample from
        num_resamples: Number of resample attempts per sentence
        model: Model to use
        api_key: OpenRouter API key
        temperature: Sampling temperature
        sentence_indices: Optional list of specific sentence indices to resample
                         (if None, resamples all sentences)
        save_results: Whether to save results to disk
        verbose: Whether to print progress

    Returns:
        Summary dictionary with all resampling results
    """
    # Initialize client
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
    )

    task = ForcedResponseTask(model=model)

    # Get cumulative CoT segments (after each sentence)
    cot_segments = get_cumulative_cot_segments(source_cot)

    if verbose:
        print(f"Found {len(cot_segments)} sentences in CoT")
        print(f"Correct answer: {question.correct_answer}")
        print(f"Resamples per sentence: {num_resamples}")
        print()

    # Determine which sentences to process
    if sentence_indices is None:
        indices_to_process = list(range(len(cot_segments)))
    else:
        indices_to_process = [i for i in sentence_indices if i < len(cot_segments)]

    all_results = []

    for sentence_idx in indices_to_process:
        forced_prefix = cot_segments[sentence_idx]

        if verbose:
            # Show first 100 chars of the prefix
            preview = forced_prefix[:100] + "..." if len(forced_prefix) > 100 else forced_prefix
            print(f"  Sentence {sentence_idx + 1}/{len(cot_segments)}: {preview}")

        resample_results = run_resampling_for_sentence(
            client=client,
            question=question,
            forced_prefix=forced_prefix,
            sentence_idx=sentence_idx,
            num_resamples=num_resamples,
            model=model,
            temperature=temperature,
            verbose=verbose,
        )

        # Save results for this sentence
        if save_results:
            task.save_resampling_result(
                question=question,
                sentence_idx=sentence_idx,
                forced_prefix=forced_prefix,
                resample_results=[r.to_dict() for r in resample_results],
            )

        # Compute summary for this sentence
        answers = [r.answer for r in resample_results if r.answer]
        valid_answers = [a for a in answers if a in ['A', 'B', 'C', 'D']]

        answer_counts = {}
        for a in valid_answers:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        total_valid = len(valid_answers)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)

        sentence_summary = {
            "sentence_idx": sentence_idx,
            "forced_prefix_length": len(forced_prefix),
            "total_resamples": len(resample_results),
            "valid_answers": total_valid,
            "answer_counts": answer_counts,
            "most_common": most_common[0],
            "most_common_count": most_common[1],
            "agreement_rate": most_common[1] / total_valid if total_valid > 0 else 0,
        }
        all_results.append(sentence_summary)

        if verbose:
            print(f"    Summary: {answer_counts}")
            print(f"    Most common: {most_common[0]} ({most_common[1]}/{total_valid})")
            print()

    # Save overall summary
    if save_results:
        task.save_resampling_summary(
            question=question,
            source_rollout_idx=0,  # Could be parameterized
            all_sentence_results=all_results,
        )

    summary = {
        "question_id": question.id,
        "correct_answer": question.correct_answer,
        "num_sentences": len(cot_segments),
        "sentences_processed": len(indices_to_process),
        "num_resamples": num_resamples,
        "sentence_results": all_results,
    }

    return summary


def run_resampling_from_verification(
    question_id: str,
    rollout_idx: int = 0,
    num_resamples: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    sentence_indices: Optional[List[int]] = None,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run resampling using a CoT from verification rollouts.

    Args:
        question_id: ID of the verified question
        rollout_idx: Which rollout to use as source (default: first)
        num_resamples: Number of resample attempts per sentence
        model: Model to use
        api_key: OpenRouter API key
        temperature: Sampling temperature
        sentence_indices: Optional list of specific sentence indices to resample
        verbose: Whether to print progress

    Returns:
        Resampling summary, or None if verification data not found
    """
    task = ForcedResponseTask(model=model)

    # Load verification data
    verification_dir = task.verification_dir / question_id
    if not verification_dir.exists():
        print(f"No verification data found for {question_id}")
        return None

    # Load the specified rollout
    rollout_path = verification_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
    if not rollout_path.exists():
        print(f"Rollout {rollout_idx} not found for {question_id}")
        return None

    with open(rollout_path) as f:
        rollout_data = json.load(f)

    # Load question from summary
    summary_path = verification_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Reconstruct question object
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
        # Fall back to full response if no separate thinking
        source_cot = rollout_data.get("full_response", "")

    if not source_cot:
        print(f"No CoT found in rollout {rollout_idx}")
        return None

    if verbose:
        print(f"Running resampling for {question_id}")
        print(f"Using rollout {rollout_idx}")
        print(f"CoT length: {len(source_cot)} characters")
        print()

    return run_resampling(
        question=question,
        source_cot=source_cot,
        num_resamples=num_resamples,
        model=model,
        api_key=api_key,
        temperature=temperature,
        sentence_indices=sentence_indices,
        verbose=verbose,
    )
