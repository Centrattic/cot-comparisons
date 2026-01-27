"""
Forcing module for the Forced Response task — true prefill forcing via Tinker.

Uses the Tinker API to prefill the start of the model's <think> block with
partial CoT, then lets the model continue its chain of thought and produce
an answer. This is "true forcing" because the model's reasoning IS seeded
with the partial CoT prefix — it continues thinking from there and answers.
"""

import asyncio
import contextlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from tinker import ServiceClient, types
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoTokenizer

from .data_loader import GPQAQuestion, BinaryJudgeQuestion, Question
from .judge import extract_outcome_from_response
from .prompts import get_cumulative_cot_segments
from .task import ForcedResponseTask

# Chat template special tokens for Kimi K2
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"


@dataclass
class ForceResult:
    """Result of a single true-forcing attempt."""

    sentence_idx: int
    force_idx: int
    partial_cot: str
    continued_cot: str
    raw_tokens: List[int]
    raw_response: str
    answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx,
            "force_idx": self.force_idx,
            "partial_cot": self.partial_cot,
            "continued_cot": self.continued_cot,
            "raw_tokens": self.raw_tokens,
            "raw_response": self.raw_response,
            "answer": self.answer,
        }


def init_tinker_client(
    model: str,
) -> Tuple[Any, AutoTokenizer]:
    """
    Initialize the Tinker sampling client and tokenizer.

    Args:
        model: HuggingFace model identifier

    Returns:
        Tuple of (sampling_client, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=model)
    return sampling_client, tokenizer


def build_force_prompt(
    question_obj: Question,
    partial_cot: str,
    tokenizer: AutoTokenizer,
) -> types.ModelInput:
    """
    Build a tokenized prompt that prefills the start of the model's thinking
    with partial CoT. The model continues its chain of thought from there,
    eventually closing </think> and producing an answer.

    The prompt follows the Kimi K2 chat template:
        <|im_system|>system<|im_middle|>{system_msg}<|im_end|>
        <|im_user|>user<|im_middle|>{user_msg}<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>{partial_cot}

    The model continues generating from the end of the partial CoT.

    Args:
        question_obj: The question (GPQAQuestion or BinaryJudgeQuestion)
        partial_cot: The partial chain of thought to prefill
        tokenizer: The tokenizer to encode with

    Returns:
        ModelInput ready for sampling
    """
    if isinstance(question_obj, BinaryJudgeQuestion):
        # Binary judge questions: just the question text, no choices
        user_msg = question_obj.question
    else:
        # Multiple choice questions: include choices
        choice_labels = [chr(ord("A") + i) for i in range(len(question_obj.choices))]
        choices_text = "\n".join(
            f"{label}. {choice}" for label, choice in zip(choice_labels, question_obj.choices)
        )
        user_msg = (
            f"{question_obj.question}\n\n{choices_text}\n\nAnswer with just the letter (A, B, C, or D)."
        )

    prompt_str = (
        f"{IM_SYSTEM}system{IM_MIDDLE}"
        f"You are Kimi, an AI assistant created by Moonshot AI."
        f"{IM_END}"
        f"{IM_USER}user{IM_MIDDLE}"
        f"{user_msg}"
        f"{IM_END}"
        f"{IM_ASSISTANT}assistant{IM_MIDDLE}"
        f"<think>{partial_cot}"
    )

    with contextlib.redirect_stdout(io.StringIO()):
        tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
    return types.ModelInput.from_ints(tokens)


def extract_force_answer_multiple_choice(
    tokens: List[int], tokenizer: AutoTokenizer
) -> Tuple[str, str]:
    """
    Decode sampled tokens and extract the answer letter from after </think>.

    The model's output will be: {continued_thinking}</think>{answer}
    We extract the answer from the content after the closing think tag.

    Args:
        tokens: List of token IDs from sampling
        tokenizer: Tokenizer for decoding

    Returns:
        Tuple of (answer_letter, full_decoded_text)
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Split on </think> to get the answer portion
    if "</think>" in text:
        parts = text.split("</think>", 1)
        answer_text = parts[1].strip()
    else:
        # Model didn't close think tag within max_tokens
        answer_text = ""

    # Try exact single letter from answer portion
    clean = answer_text.upper().rstrip(".").strip()
    if clean in ["A", "B", "C", "D"]:
        return clean, text

    # Try to find a letter in the answer portion
    match = re.search(r"\b([A-Da-d])\b", answer_text)
    if match:
        return match.group(1).upper(), text

    return "", text


def extract_force_answer(
    tokens: List[int],
    tokenizer: AutoTokenizer,
    question_obj: Question,
) -> Tuple[str, str, bool]:
    """
    Decode sampled tokens and extract the answer based on question type.

    For multiple_choice: extracts A/B/C/D from after </think>
    For binary_judge: calls LLM judge to classify the full response

    Args:
        tokens: List of token IDs from sampling
        tokenizer: Tokenizer for decoding
        question_obj: The question (determines extraction method)

    Returns:
        Tuple of (answer, full_decoded_text, is_valid)
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    if isinstance(question_obj, BinaryJudgeQuestion):
        # Binary judge: get the full response (including any tool use)
        # and call the LLM judge to classify it
        outcome, is_valid = extract_outcome_from_response(
            response=text,
            question_type="binary_judge",
            judge_prompt=question_obj.judge_prompt,
            bad_outcome=question_obj.bad_outcome,
        )
        return outcome, text, is_valid
    else:
        # Multiple choice: extract A/B/C/D
        answer, decoded = extract_force_answer_multiple_choice(tokens, tokenizer)
        is_valid = answer != ""
        return answer, decoded, is_valid


def run_forcing(
    question: Question,
    source_cot: str,
    model: str,
    num_forces: int = 5,
    max_sentences: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    save_results: bool = True,
    verbose: bool = True,
    rollout_idx: int = 0,
) -> Dict[str, Any]:
    """
    Run true forcing for all sentences in the CoT.

    For each sentence boundary, prefills the model's thinking with the partial
    CoT up to that point, then lets the model continue its chain of thought
    and produce an answer. All num_forces samples per sentence are obtained
    in a single Tinker call.

    Args:
        question: The GPQA question
        source_cot: The full chain of thought
        num_forces: Number of force samples per sentence
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Max tokens to generate (model finishes thinking + answer)
        save_results: Whether to save results to disk
        verbose: Whether to print progress
        rollout_idx: Index of the source rollout being forced

    Returns:
        Summary dictionary with all forcing results
    """
    sampling_client, tokenizer = init_tinker_client(model)
    task = ForcedResponseTask(model=model)

    # Get the im_end token ID for stop sequence
    with contextlib.redirect_stdout(io.StringIO()):
        im_end_token = tokenizer.encode(IM_END, add_special_tokens=False)

    # Get cumulative CoT segments (after each sentence)
    cot_segments = get_cumulative_cot_segments(source_cot)
    if max_sentences is not None:
        cot_segments = cot_segments[:max_sentences]
    num_sentences = len(cot_segments)

    # Create timestamped run directory
    run_dir = None
    if save_results:
        config = {
            "model": model,
            "question_id": question.id,
            "rollout_idx": rollout_idx,
            "num_forces": num_forces,
            "max_sentences": max_sentences,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_sentences": num_sentences,
        }
        run_dir = task.create_run_dir("forcing", question.id, rollout_idx, config)

    if verbose:
        print(f"Found {num_sentences} sentences in CoT")
        if isinstance(question, GPQAQuestion):
            print(f"Correct answer: {question.correct_answer}")
        else:
            print(f"Question type: binary_judge (bad outcome: {question.bad_outcome})")
        print(f"Running {num_forces} forces per sentence via Tinker")
        if run_dir:
            print(f"Run dir: {run_dir}")
        print()

    # Tokenize full source CoT once to compute per-sentence max_tokens
    with contextlib.redirect_stdout(io.StringIO()):
        source_cot_token_count = len(
            tokenizer.encode(source_cot, add_special_tokens=False)
        )

    def force_single_sample(
        sentence_idx: int, force_idx: int, partial_cot: str
    ) -> Optional[ForceResult]:
        """Force a single sample - runs in thread pool. Returns ForceResult or None."""
        # max_tokens = remaining CoT tokens + buffer, capped at 1024
        remaining_frac = 1.0 - len(partial_cot) / max(len(source_cot), 1)
        sentence_max_tokens = int(source_cot_token_count * remaining_frac) + 100
        sentence_max_tokens = min(max(sentence_max_tokens, 100), 1024)

        params = types.SamplingParams(
            max_tokens=sentence_max_tokens,
            temperature=temperature,
        )

        prompt = build_force_prompt(
            question_obj=question,
            partial_cot=partial_cot,
            tokenizer=tokenizer,
        )

        # num_samples=1 like friend's code
        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        ).result()

        sample_tokens = result.sequences[0].tokens
        answer, raw_response, is_valid = extract_force_answer(
            sample_tokens, tokenizer, question
        )

        if is_valid and answer:
            if "</think>" in raw_response:
                continued_cot = raw_response.split("</think>", 1)[0]
            else:
                continued_cot = raw_response

            return ForceResult(
                sentence_idx=sentence_idx,
                force_idx=force_idx,
                partial_cot=partial_cot,
                continued_cot=continued_cot,
                raw_tokens=list(sample_tokens),
                raw_response=raw_response,
                answer=answer,
            )
        return None

    # Build all (sentence_idx, force_idx) tasks
    all_tasks = [
        (sent_idx, force_idx, cot_segments[sent_idx])
        for sent_idx in range(num_sentences)
        for force_idx in range(num_forces)
    ]
    total_tasks = len(all_tasks)

    if verbose:
        print(
            f"Submitting {total_tasks} parallel jobs ({num_sentences} sentences × {num_forces} forces)"
        )

    # Run ALL samples in parallel with ThreadPoolExecutor
    all_results: List[ForceResult] = []
    num_workers = min(300, total_tasks)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(force_single_sample, sent_idx, force_idx, partial_cot): (
                sent_idx,
                force_idx,
            )
            for sent_idx, force_idx, partial_cot in all_tasks
        }

        with tqdm(
            total=total_tasks, desc="Forcing", unit="sample", disable=not verbose
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                pbar.update(1)

    # Group results by sentence
    results_by_sentence: Dict[int, List[ForceResult]] = {}
    for r in all_results:
        if r.sentence_idx not in results_by_sentence:
            results_by_sentence[r.sentence_idx] = []
        results_by_sentence[r.sentence_idx].append(r)

    # Build summaries and save
    all_sentence_summaries = []
    for sent_idx in range(num_sentences):
        sent_results = results_by_sentence.get(sent_idx, [])
        partial_cot = cot_segments[sent_idx]

        if save_results:
            task.save_forcing_result(
                question=question,
                sentence_idx=sent_idx,
                partial_cot=partial_cot,
                force_results=[r.to_dict() for r in sent_results],
                rollout_idx=rollout_idx,
                run_dir=run_dir,
            )

        answer_counts: Dict[str, int] = {}
        for r in sent_results:
            answer_counts[r.answer] = answer_counts.get(r.answer, 0) + 1

        all_sentence_summaries.append(
            {
                "sentence_idx": sent_idx,
                "partial_cot_length": len(partial_cot),
                "total_forces": len(sent_results),
                "valid_answers": len(sent_results),
                "answer_counts": answer_counts,
                "most_common": max(answer_counts.items(), key=lambda x: x[1])[0]
                if answer_counts
                else "",
            }
        )

    # Save overall summary
    if save_results:
        task.save_forcing_summary(
            question=question,
            source_rollout_idx=rollout_idx,
            all_sentence_results=all_sentence_summaries,
            source_cot=source_cot,
            run_dir=run_dir,
        )

    if verbose:
        print(f"\nDone. Processed {num_sentences} sentences.")

    result = {
        "question_id": question.id,
        "question_type": question.question_type,
        "num_sentences": num_sentences,
        "sentence_results": all_sentence_summaries,
    }
    if isinstance(question, GPQAQuestion):
        result["correct_answer"] = question.correct_answer
    else:
        result["bad_outcome"] = question.bad_outcome
    return result


def run_forcing_from_verification(
    question_id: str,
    model: str,
    rollout_idx: int = 0,
    num_forces: int = 5,
    max_sentences: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run true forcing using a CoT from verification rollouts.

    Args:
        question_id: ID of the question to force
        rollout_idx: Which verification rollout to use as source
        num_forces: Number of force samples per sentence
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Max tokens to generate (model finishes thinking + answer)
        verbose: Whether to print progress

    Returns:
        Summary dictionary or None if data not found
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

    # Create the appropriate question type
    question_type = summary.get("question_type", "multiple_choice")
    if question_type == "binary_judge":
        question: Question = BinaryJudgeQuestion(
            id=summary["question_id"],
            question=summary["question"],
            judge_prompt=summary["judge_prompt"],
            bad_outcome=summary["bad_outcome"],
            subject=summary.get("subject"),
        )
    else:
        question = GPQAQuestion(
            id=summary["question_id"],
            question=summary["question"],
            choices=summary["choices"],
            correct_answer=summary["correct_answer"],
            correct_index=ord(summary["correct_answer"]) - ord("A"),
        )

    # Get the CoT from the rollout
    source_cot = rollout_data.get("thinking", "")
    if not source_cot:
        source_cot = rollout_data.get("full_response", "")

    if not source_cot:
        print(f"No CoT found in rollout {rollout_idx}")
        return None

    if verbose:
        print(f"Running forcing (Tinker) for {question_id}")
        print(f"Using rollout {rollout_idx}")
        print(f"CoT length: {len(source_cot)} characters")
        print()

    return run_forcing(
        question=question,
        source_cot=source_cot,
        num_forces=num_forces,
        max_sentences=max_sentences,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
        rollout_idx=rollout_idx,
    )
