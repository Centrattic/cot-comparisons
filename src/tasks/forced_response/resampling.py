"""
Resampling module — Tinker-based prefix continuation.

Takes ~20 evenly-spaced prefixes of a CoT, forces each as the start of
the model's <think> block, and lets the model generate 20 continuations
per prefix. This gives the resampled answer distribution at each point.
"""

import io
import json
import re
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoTokenizer
from tinker import ServiceClient, types

from .data_loader import GPQAQuestion
from .prompts import get_cumulative_cot_segments
from .task import ForcedResponseTask


# Kimi K2 chat template tokens
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"


@dataclass
class ResampleResult:
    """Result of a single resample attempt."""
    sentence_idx: int
    resample_idx: int
    forced_prefix: str
    continuation: str
    full_response: str
    answer: str
    raw_tokens: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx,
            "resample_idx": self.resample_idx,
            "forced_prefix": self.forced_prefix,
            "continuation": self.continuation,
            "full_response": self.full_response,
            "answer": self.answer,
            "raw_tokens": self.raw_tokens,
        }


def init_tinker_client(model: str) -> Tuple[Any, AutoTokenizer]:
    """Initialize the Tinker sampling client and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=model)
    return sampling_client, tokenizer


def build_resample_prompt(
    question: str,
    choices: List[str],
    partial_cot: str,
    tokenizer: AutoTokenizer,
) -> types.ModelInput:
    """
    Build a tokenized prompt that prefills the start of the model's thinking
    with partial CoT. The model continues generating from there.
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


def extract_resample_answer(tokens: List[int], tokenizer: AutoTokenizer) -> Tuple[str, str, str]:
    """
    Decode sampled tokens and extract the answer from after </think>.

    Returns:
        Tuple of (answer_letter, continued_cot, full_decoded_text)
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    if "</think>" in text:
        parts = text.split("</think>", 1)
        continued_cot = parts[0]
        answer_text = parts[1].strip()
    else:
        continued_cot = text
        answer_text = ""

    # Extract answer letter
    clean = answer_text.upper().rstrip('.').strip()
    if clean in ['A', 'B', 'C', 'D']:
        return clean, continued_cot, text

    match = re.search(r'\b([A-Da-d])\b', answer_text)
    if match:
        return match.group(1).upper(), continued_cot, text

    return "", continued_cot, text


def run_resampling_from_verification(
    question_id: str,
    model: str,
    rollout_idx: int = 0,
    num_resamples: int = 20,
    num_prefix_points: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run resampling using a CoT from verification rollouts.

    Selects ~num_prefix_points evenly-spaced prefix points from the CoT,
    then runs num_resamples continuations at each point via Tinker.

    Args:
        question_id: ID of the question to resample
        model: HuggingFace model identifier for Tinker
        rollout_idx: Which verification rollout to use as source
        num_resamples: Number of continuations per prefix point
        num_prefix_points: Target number of prefix points (~20)
        temperature: Sampling temperature
        max_tokens: Max tokens to generate per continuation
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

    # Get cumulative CoT segments
    cot_segments = get_cumulative_cot_segments(source_cot)
    num_sentences = len(cot_segments)

    # Compute stride for ~num_prefix_points evenly-spaced points
    stride = max(num_sentences // num_prefix_points, 1)
    selected_indices = list(range(0, num_sentences, stride))

    if verbose:
        print(f"Running resampling (Tinker) for {question_id}")
        print(f"Using rollout {rollout_idx}")
        print(f"CoT length: {len(source_cot)} characters")
        print(f"Total sentences: {num_sentences}, stride: {stride}, prefix points: {len(selected_indices)}")
        print(f"Resamples per point: {num_resamples}")
        print()

    # Initialize Tinker
    sampling_client, tokenizer = init_tinker_client(model)
    with contextlib.redirect_stdout(io.StringIO()):
        im_end_token = tokenizer.encode(IM_END, add_special_tokens=False)

    # Create timestamped run directory
    config = {
        "model": model,
        "question_id": question.id,
        "rollout_idx": rollout_idx,
        "num_resamples": num_resamples,
        "num_prefix_points": len(selected_indices),
        "stride": stride,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_sentences": num_sentences,
    }
    run_dir = task.create_run_dir("resampling", question.id, rollout_idx, config)
    if verbose:
        print(f"Run dir: {run_dir}")
        print()

    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    def resample_single(sentence_idx: int, resample_idx: int, partial_cot: str) -> Optional[ResampleResult]:
        """Run a single resample - one job per (sentence, resample) pair."""
        prompt = build_resample_prompt(
            question=question.question,
            choices=question.choices,
            partial_cot=partial_cot,
            tokenizer=tokenizer,
        )

        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        ).result()

        sample_tokens = result.sequences[0].tokens
        answer, continued_cot, full_text = extract_resample_answer(sample_tokens, tokenizer)

        if answer:
            return ResampleResult(
                sentence_idx=sentence_idx,
                resample_idx=resample_idx,
                forced_prefix=partial_cot,
                continuation=continued_cot,
                full_response=partial_cot + full_text,
                answer=answer,
                raw_tokens=list(sample_tokens),
            )
        return None

    # Build all (sentence_idx, resample_idx) tasks
    all_tasks = [
        (sent_idx, resample_idx, cot_segments[sent_idx])
        for sent_idx in selected_indices
        for resample_idx in range(num_resamples)
    ]
    total_tasks = len(all_tasks)

    if verbose:
        print(f"Submitting {total_tasks} parallel jobs ({len(selected_indices)} prefixes × {num_resamples} resamples)")

    # Run ALL resamples in parallel with ThreadPoolExecutor
    all_results: List[ResampleResult] = []
    num_workers = min(300, total_tasks)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(resample_single, sent_idx, resample_idx, partial_cot): (sent_idx, resample_idx)
            for sent_idx, resample_idx, partial_cot in all_tasks
        }

        with tqdm(total=total_tasks, desc="Resampling", unit="sample", disable=not verbose) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                pbar.update(1)

    # Group results by sentence
    results_by_sentence: Dict[int, List[ResampleResult]] = {}
    for r in all_results:
        if r.sentence_idx not in results_by_sentence:
            results_by_sentence[r.sentence_idx] = []
        results_by_sentence[r.sentence_idx].append(r)

    # Build summaries and save
    all_sentence_summaries = []
    for sent_idx in selected_indices:
        sent_results = results_by_sentence.get(sent_idx, [])
        partial_cot = cot_segments[sent_idx]

        # Save per-prefix results
        task.save_resampling_result(
            question=question,
            sentence_idx=sent_idx,
            forced_prefix=partial_cot,
            resample_results=[r.to_dict() for r in sent_results],
            rollout_idx=rollout_idx,
            run_dir=run_dir,
        )

        # Compute summary for this prefix point
        answer_counts: Dict[str, int] = {}
        for r in sent_results:
            answer_counts[r.answer] = answer_counts.get(r.answer, 0) + 1

        total_valid = len(sent_results)
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)

        sentence_summary = {
            "sentence_idx": sent_idx,
            "forced_prefix_length": len(partial_cot),
            "total_resamples": total_valid,
            "valid_answers": total_valid,
            "answer_counts": answer_counts,
            "most_common": most_common[0],
            "most_common_count": most_common[1],
            "agreement_rate": most_common[1] / total_valid if total_valid > 0 else 0,
        }
        all_sentence_summaries.append(sentence_summary)

    # Save overall summary
    task.save_resampling_summary(
        question=question,
        source_rollout_idx=rollout_idx,
        all_sentence_results=all_sentence_summaries,
        run_dir=run_dir,
    )

    if verbose:
        print(f"\nDone. Processed {len(selected_indices)} prefix points.")

    return {
        "question_id": question.id,
        "correct_answer": question.correct_answer,
        "num_sentences": num_sentences,
        "num_prefix_points": len(selected_indices),
        "stride": stride,
        "num_resamples": num_resamples,
        "sentence_results": all_sentence_summaries,
    }
