"""
Shared verification utility — generate free-rollout CoT data via Tinker.

Produces verification data consumed by ForcingTask and ResamplingTask.
Both tasks read from verification/<question_id>/<timestamp>/ directories
via their load_question_and_cot() methods.

Usage:
    from src2.utils.verification import run_verification, ensure_verification

    summary = run_verification(question, num_rollouts=50, model="moonshotai/kimi-k2-thinking",
                               verification_dir=Path("data/forced_response/verification"))

    # Or the convenience wrapper that skips if data already exists:
    summary = ensure_verification(question, verification_dir=..., model=...)
"""

import contextlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .questions import GPQAQuestion, BinaryJudgeQuestion, Question


# Kimi K2 chat template tokens
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"


@dataclass
class VerificationRollout:
    """Result of a single free-rollout verification attempt."""
    rollout_idx: int
    question_id: str
    full_prompt: str       # Exact prompt string with all chat template tags
    thinking: str          # Text between <think> and </think>
    answer: str            # Extracted answer: A/B/C/D or YES/NO
    full_response: str     # Raw decoded continuation from Tinker

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_idx": self.rollout_idx,
            "full_prompt": self.full_prompt,
            "thinking": self.thinking,
            "answer": self.answer,
            "full_response": self.full_response,
        }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_user_message(question: Question) -> str:
    """Format a Question into the user message text."""
    if isinstance(question, BinaryJudgeQuestion):
        return question.question
    labels = [chr(ord("A") + i) for i in range(len(question.choices))]
    choices = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
    return f"{question.question}\n\n{choices}\n\nAnswer with just the letter (A, B, C, or D)."


def build_verification_prompt(question: Question) -> str:
    """
    Build the full Tinker prompt for a free verification rollout.

    The prompt includes system/user/assistant chat template tags and ends
    with ``<think>`` so the model generates its chain of thought freely.
    The returned string is saved as ``full_prompt`` in each rollout JSON
    so activations can later be extracted over the exact input.
    """
    user_msg = _format_user_message(question)
    return (
        f"{IM_SYSTEM}system{IM_MIDDLE}You are Kimi, an AI assistant created by Moonshot AI.{IM_END}"
        f"{IM_USER}user{IM_MIDDLE}{user_msg}{IM_END}"
        f"{IM_ASSISTANT}assistant{IM_MIDDLE}<think>"
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer(raw_response: str, question: Question) -> str:
    """
    Extract the final answer from a decoded Tinker continuation.

    For GPQAQuestion: looks for A/B/C/D after </think>.
    For BinaryJudgeQuestion: calls the LLM judge.
    """
    if isinstance(question, BinaryJudgeQuestion):
        # Check for explicit YES/NO after </think>
        if "</think>" in raw_response:
            after = raw_response.split("</think>", 1)[1].strip().upper()
            if "YES" in after:
                return "YES"
            if "NO" in after:
                return "NO"
        # Fall back to LLM judge
        from .judge import extract_binary_answer
        return extract_binary_answer(
            raw_response, question.judge_prompt, question.bad_outcome,
        )

    # Multiple choice — look after </think>
    if "</think>" in raw_response:
        after = raw_response.split("</think>", 1)[1].strip().upper().rstrip(".")
        if after in ["A", "B", "C", "D"]:
            return after
        match = re.search(r"\b([A-D])\b", after)
        if match:
            return match.group(1)
    return ""


def _extract_thinking(raw_response: str) -> str:
    """Extract the CoT text between <think> and </think>."""
    if "</think>" in raw_response:
        return raw_response.split("</think>", 1)[0]
    return raw_response


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_single_rollout(
    question: Question,
    rollout_idx: int,
    sampling_client,
    tokenizer,
    full_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> Optional[VerificationRollout]:
    """
    Run one free rollout via Tinker and extract the answer.

    Args:
        question: The question to verify.
        rollout_idx: Index of this rollout.
        sampling_client: Tinker SamplingClient.
        tokenizer: HuggingFace tokenizer for the model.
        full_prompt: Pre-built prompt from build_verification_prompt().
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        VerificationRollout if a valid answer was extracted, None otherwise.
    """
    from tinker import types

    with contextlib.redirect_stdout(io.StringIO()):
        tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    model_input = types.ModelInput.from_ints(tokens)

    params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
    result = sampling_client.sample(
        prompt=model_input, num_samples=1, sampling_params=params,
    ).result()

    sample_tokens = result.sequences[0].tokens
    raw_response = tokenizer.decode(sample_tokens, skip_special_tokens=True)

    answer = _extract_answer(raw_response, question)
    thinking = _extract_thinking(raw_response)

    return VerificationRollout(
        rollout_idx=rollout_idx,
        question_id=question.id,
        full_prompt=full_prompt,
        thinking=thinking,
        answer=answer,
        full_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(question: Question, rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute agreement statistics from rollout results."""
    answers = [r["answer"].upper() for r in rollouts if r.get("answer")]

    if isinstance(question, BinaryJudgeQuestion):
        valid_answers = [a for a in answers if a in ["YES", "NO"]]
    else:
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]

    answer_counts: Dict[str, int] = {}
    for a in valid_answers:
        answer_counts[a] = answer_counts.get(a, 0) + 1

    total_valid = len(valid_answers)
    most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)
    agreement_rate = most_common[1] / total_valid if total_valid > 0 else 0.0

    summary: Dict[str, Any] = {
        "question_id": question.id,
        "question": question.question,
        "question_type": question.question_type,
        "total_rollouts": len(rollouts),
        "valid_rollouts": total_valid,
        "answer_counts": answer_counts,
        "most_common_answer": most_common[0],
        "most_common_count": most_common[1],
        "agreement_rate": agreement_rate,
        "meets_threshold": agreement_rate >= 0.8,
    }

    if isinstance(question, GPQAQuestion):
        summary["choices"] = question.choices
        summary["correct_answer"] = question.correct_answer
        summary["is_correct"] = most_common[0] == question.correct_answer
    elif isinstance(question, BinaryJudgeQuestion):
        summary["judge_prompt"] = question.judge_prompt
        summary["bad_outcome"] = question.bad_outcome
        summary["bad_outcome_rate"] = (
            answer_counts.get(question.bad_outcome.upper(), 0) / total_valid
            if total_valid > 0 else 0.0
        )
        if question.subject:
            summary["subject"] = question.subject

    return summary


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _get_latest_verification_dir(verification_dir: Path, question_id: str) -> Optional[Path]:
    """Find the latest timestamped run directory for a question."""
    qdir = verification_dir / question_id
    if not qdir.exists():
        return None
    timestamped = sorted(
        [d for d in qdir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"],
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    if (qdir / "summary.json").exists():
        return qdir
    return None


def _load_existing_summary(verification_dir: Path, question_id: str) -> Optional[Dict[str, Any]]:
    """Load the summary from the latest verification run, if it exists."""
    run_dir = _get_latest_verification_dir(verification_dir, question_id)
    if run_dir is None:
        return None
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_verification(
    question: Question,
    num_rollouts: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    verification_dir: Optional[Path] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_workers: int = 300,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run verification rollouts for a single question via Tinker.

    Generates ``num_rollouts`` free rollouts, computes agreement statistics,
    and saves everything to ``verification_dir/<question_id>/<timestamp>/``.

    Args:
        question: The question to verify.
        num_rollouts: Number of independent rollouts to generate.
        model: Tinker model identifier.
        verification_dir: Root directory for verification data.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per rollout.
        max_workers: Maximum concurrent Tinker requests.
        verbose: Print progress.

    Returns:
        Summary dict with agreement statistics.
    """
    from tinker import ServiceClient
    from transformers import AutoTokenizer

    if verification_dir is None:
        verification_dir = (
            Path(__file__).parent.parent.parent / "data" / "forced_response" / "verification"
        )

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = verification_dir / question.id / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)

    # Save config
    config = {
        "model": model,
        "question_id": question.id,
        "question_type": question.question_type,
        "num_rollouts": num_rollouts,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_workers": max_workers,
        "timestamp": datetime.now().isoformat(),
        "run_type": "verification",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Initialize Tinker
    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=model)

    full_prompt = build_verification_prompt(question)

    if verbose:
        print(f"Verifying {question.id}: {num_rollouts} rollouts (model={model})")
        print(f"Run dir: {run_dir}")

    # Run rollouts in parallel
    all_rollouts: List[Optional[VerificationRollout]] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, num_rollouts)) as executor:
        futures = {
            executor.submit(
                run_single_rollout,
                question, idx, sampling_client, tokenizer,
                full_prompt, temperature, max_tokens,
            ): idx
            for idx in range(num_rollouts)
        }
        for future in tqdm(
            as_completed(futures), total=num_rollouts,
            desc="Verification", disable=not verbose,
        ):
            idx = futures[future]
            try:
                rollout = future.result()
                all_rollouts.append(rollout)
            except Exception as e:
                if verbose:
                    print(f"  Rollout {idx} failed: {e}")
                all_rollouts.append(None)

    # Save individual rollouts
    rollout_dicts = []
    for rollout in all_rollouts:
        if rollout is None:
            continue
        d = rollout.to_dict()
        rollout_dicts.append(d)
        path = rollouts_dir / f"rollout_{rollout.rollout_idx:03d}.json"
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    # Compute and save summary
    summary = compute_summary(question, rollout_dicts)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        rate = summary["agreement_rate"]
        top = summary["most_common_answer"]
        print(
            f"  {question.id}: {summary['valid_rollouts']}/{summary['total_rollouts']} valid, "
            f"agreement={rate:.0%} (most common: {top})"
        )

    return summary


def ensure_verification(
    question: Question,
    verification_dir: Path,
    num_rollouts: int = 50,
    model: str = "moonshotai/kimi-k2-thinking",
    threshold: float = 0.8,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Ensure verification data exists for a question.

    If a previous run exists and meets the agreement threshold, returns
    its summary without re-running. Otherwise runs verification.

    Args:
        question: The question to verify.
        verification_dir: Root directory for verification data.
        num_rollouts: Number of rollouts (only used if running fresh).
        model: Tinker model identifier.
        threshold: Minimum agreement rate to accept existing data.
        **kwargs: Passed through to run_verification().

    Returns:
        Summary dict if verification succeeded, None if below threshold.
    """
    existing = _load_existing_summary(verification_dir, question.id)
    if existing is not None and existing.get("agreement_rate", 0) >= threshold:
        return existing

    summary = run_verification(
        question=question,
        num_rollouts=num_rollouts,
        model=model,
        verification_dir=verification_dir,
        **kwargs,
    )

    if summary.get("agreement_rate", 0) >= threshold:
        return summary
    return None
