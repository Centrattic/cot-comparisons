"""
Forcing module for the Forced Response task — true prefill forcing via Tinker.

Uses the Tinker API to prefill the start of the model's <think> block with
partial CoT, then lets the model continue its chain of thought and produce
an answer. This is "true forcing" because the model's reasoning IS seeded
with the partial CoT prefix — it continues thinking from there and answers.
"""

import io
import json
import os
import re
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoTokenizer
from tinker import ServiceClient, types

from .data_loader import GPQAQuestion
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
    question: str,
    choices: List[str],
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
        question: The question text
        choices: List of answer choices
        partial_cot: The partial chain of thought to prefill
        tokenizer: The tokenizer to encode with

    Returns:
        ModelInput ready for sampling
    """
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    choices_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(choice_labels, choices)
    )

    user_msg = (
        f"{question}\n\n{choices_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
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


def extract_force_answer(tokens: List[int], tokenizer: AutoTokenizer) -> Tuple[str, str]:
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
    clean = answer_text.upper().rstrip('.').strip()
    if clean in ['A', 'B', 'C', 'D']:
        return clean, text

    # Try to find a letter in the answer portion
    match = re.search(r'\b([A-Da-d])\b', answer_text)
    if match:
        return match.group(1).upper(), text

    return "", text


def run_forcing(
    question: GPQAQuestion,
    source_cot: str,
    model: str,
    num_forces: int = 5,
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
    num_sentences = len(cot_segments)

    # Create timestamped run directory
    run_dir = None
    if save_results:
        config = {
            "model": model,
            "question_id": question.id,
            "rollout_idx": rollout_idx,
            "num_forces": num_forces,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_sentences": num_sentences,
        }
        run_dir = task.create_run_dir("forcing", question.id, rollout_idx, config)

    if verbose:
        print(f"Found {num_sentences} sentences in CoT")
        print(f"Correct answer: {question.correct_answer}")
        print(f"Running {num_forces} forces per sentence via Tinker")
        if run_dir:
            print(f"Run dir: {run_dir}")
        print()

    all_sentence_summaries = []

    # Tokenize full source CoT once to compute per-sentence max_tokens
    with contextlib.redirect_stdout(io.StringIO()):
        source_cot_token_count = len(tokenizer.encode(source_cot, add_special_tokens=False))

    # Submit all sampling requests upfront for maximum parallelism
    futures = []
    for sentence_idx, partial_cot in enumerate(cot_segments):
        # max_tokens = remaining CoT tokens + buffer for </think> and answer
        remaining_frac = 1.0 - len(partial_cot) / max(len(source_cot), 1)
        sentence_max_tokens = int(source_cot_token_count * remaining_frac) + 100
        sentence_max_tokens = max(sentence_max_tokens, 100)

        params = types.SamplingParams(
            max_tokens=sentence_max_tokens,
            temperature=temperature,
            stop=im_end_token,
        )

        prompt = build_force_prompt(
            question=question.question,
            choices=question.choices,
            partial_cot=partial_cot,
            tokenizer=tokenizer,
        )
        future = sampling_client.sample(
            prompt=prompt,
            num_samples=num_forces,
            sampling_params=params,
        )
        futures.append((sentence_idx, partial_cot, params, future))

    # Collect results with progress bar
    for sentence_idx, partial_cot, params, future in tqdm(
        futures, desc="Forcing", unit="sentence", disable=not verbose
    ):
        valid_results: List[ForceResult] = []
        result = future.result()

        for i in range(len(result.sequences)):
            sample_tokens = result.sequences[i].tokens
            answer, raw_response = extract_force_answer(sample_tokens, tokenizer)

            if answer:
                if "</think>" in raw_response:
                    continued_cot = raw_response.split("</think>", 1)[0]
                else:
                    continued_cot = raw_response

                force_result = ForceResult(
                    sentence_idx=sentence_idx,
                    force_idx=len(valid_results),
                    partial_cot=partial_cot,
                    continued_cot=continued_cot,
                    raw_tokens=list(sample_tokens),
                    raw_response=raw_response,
                    answer=answer,
                )
                valid_results.append(force_result)

        # One retry if not enough valid answers
        if len(valid_results) < num_forces:
            needed = num_forces - len(valid_results)
            prompt = build_force_prompt(
                question=question.question,
                choices=question.choices,
                partial_cot=partial_cot,
                tokenizer=tokenizer,
            )
            retry_result = sampling_client.sample(
                prompt=prompt,
                num_samples=needed,
                sampling_params=params,
            ).result()

            for i in range(len(retry_result.sequences)):
                sample_tokens = retry_result.sequences[i].tokens
                answer, raw_response = extract_force_answer(sample_tokens, tokenizer)

                if answer:
                    if "</think>" in raw_response:
                        continued_cot = raw_response.split("</think>", 1)[0]
                    else:
                        continued_cot = raw_response

                    force_result = ForceResult(
                        sentence_idx=sentence_idx,
                        force_idx=len(valid_results),
                        partial_cot=partial_cot,
                        continued_cot=continued_cot,
                        raw_tokens=list(sample_tokens),
                        raw_response=raw_response,
                        answer=answer,
                    )
                    valid_results.append(force_result)

        # Save per-sentence results
        if save_results:
            task.save_forcing_result(
                question=question,
                sentence_idx=sentence_idx,
                partial_cot=partial_cot,
                force_results=[r.to_dict() for r in valid_results],
                rollout_idx=rollout_idx,
                run_dir=run_dir,
            )

        # Compute sentence summary
        answer_counts: Dict[str, int] = {}
        for r in valid_results:
            answer_counts[r.answer] = answer_counts.get(r.answer, 0) + 1

        sentence_summary = {
            "sentence_idx": sentence_idx,
            "partial_cot_length": len(partial_cot),
            "total_forces": len(valid_results),
            "valid_answers": len(valid_results),
            "answer_counts": answer_counts,
            "most_common": max(answer_counts.items(), key=lambda x: x[1])[0] if answer_counts else "",
        }
        all_sentence_summaries.append(sentence_summary)

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

    return {
        "question_id": question.id,
        "correct_answer": question.correct_answer,
        "num_sentences": num_sentences,
        "sentence_results": all_sentence_summaries,
    }


def run_forcing_from_verification(
    question_id: str,
    model: str,
    rollout_idx: int = 0,
    num_forces: int = 5,
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
        print(f"Running forcing (Tinker) for {question_id}")
        print(f"Using rollout {rollout_idx}")
        print(f"CoT length: {len(source_cot)} characters")
        print()

    return run_forcing(
        question=question,
        source_cot=source_cot,
        num_forces=num_forces,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
        rollout_idx=rollout_idx,
    )
