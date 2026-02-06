#!/usr/bin/env python3
"""
Compare two forcing strategies using next-token logprobs.

Instead of temperature-0 argmax, we look at the actual model distribution
over answer choices (A/B/C/D) at the forcing point. Two strategies:

  1. "end_think": Force </think> after partial CoT, read the distribution
     over the first response token.

  2. "cot_anchor": Force "So, the answer is: " inside the <think> block,
     read the distribution over the next token.

We generate max_tokens=1 and read the model's logprob for the sampled token
from SampledSequence.logprobs. To get the *full* distribution over A/B/C/D
we also use topk_prompt_logprobs with a dummy appended token — this gives
us the top-k at the generation position in a single call.

Usage:
    python -m src2.runs.run_logprob_forcing
    python -m src2.runs.run_logprob_forcing --question_id starfish
    python -m src2.runs.run_logprob_forcing --question_id starfish --sentence_stride 5
"""

import argparse
import contextlib
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src2.tasks.forced_response.prompts import get_cumulative_cot_segments
from src2.utils.chat_template import build_thinking_prompt
from src2.utils.questions import load_custom_questions, GPQAQuestion, BinaryJudgeQuestion

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
CUSTOM_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "utils" / "questions.json"

MC_CHOICES = ["A", "B", "C", "D"]
BINARY_CHOICES = ["YES", "NO"]

# How many top tokens to request at the generation position
TOPK = 20

# Anchor phrases to inject inside the CoT before </think>
COT_ANCHORS = [
    "So, the answer is: ",
    "Therefore, the correct answer is ",
    "Based on my analysis, the answer is ",
]


# ── Helpers ───────────────────────────────────────────────────────────

def _user_msg(question) -> str:
    """Build the user-facing question string (same as ForcingTask._user_msg)."""
    if isinstance(question, BinaryJudgeQuestion):
        return question.question
    labels = [chr(ord("A") + i) for i in range(len(question.choices))]
    choices = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
    return f"{question.question}\n\n{choices}\n\nAnswer with just the letter (A, B, C, or D)."


def get_answer_choices(question) -> List[str]:
    """Return candidate answer tokens for this question type."""
    if isinstance(question, BinaryJudgeQuestion):
        return BINARY_CHOICES
    return MC_CHOICES


def get_latest_verification_dir(question_id: str) -> Optional[Path]:
    """Find the most recent verification run directory for a question."""
    qdir = VERIFICATION_DIR / question_id
    if not qdir.exists():
        return None
    timestamped = sorted(
        [d for d in qdir.iterdir()
         if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"],
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    if (qdir / "summary.json").exists():
        return qdir
    return None


def load_question_and_cot(question_id: str, rollout_idx: int = 0):
    """Load a question object and its source CoT from verification data."""
    run_dir = get_latest_verification_dir(question_id)
    if run_dir is None:
        return None

    rollout_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
    summary_path = run_dir / "summary.json"
    if not rollout_path.exists() or not summary_path.exists():
        return None

    with open(rollout_path) as f:
        rollout_data = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    qt = summary.get("question_type", "multiple_choice")
    if qt == "binary_judge":
        question = BinaryJudgeQuestion(
            id=summary["question_id"], question=summary["question"],
            judge_prompt=summary["judge_prompt"], bad_outcome=summary["bad_outcome"],
            subject=summary.get("subject"),
        )
    else:
        question = GPQAQuestion(
            id=summary["question_id"], question=summary["question"],
            choices=summary["choices"], correct_answer=summary["correct_answer"],
            correct_index=ord(summary["correct_answer"]) - ord("A"),
        )

    source_cot = rollout_data.get("thinking", "") or rollout_data.get("full_response", "")
    if not source_cot:
        return None
    return question, source_cot


# ── Core: logprob extraction ─────────────────────────────────────────

def _resolve_choice_token_ids(tokenizer, choices: List[str]) -> Dict[str, int]:
    """Map each answer string (e.g. "A") to its single token id."""
    mapping = {}
    for c in choices:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = tokenizer.encode(c, add_special_tokens=False)
        # Use the last token in case the tokenizer adds a prefix token
        mapping[c] = ids[-1]
    return mapping


def get_choice_logprobs(
    sampling_client,
    tokenizer,
    prompt_str: str,
    choices: List[str],
    choice_token_ids: Dict[str, int],
    types_module,
    topk: int = TOPK,
) -> Dict:
    """
    Get the next-token distribution at the generation position.

    1. Generate max_tokens=1 to get the sampled token + its logprob
       (from SampledSequence.logprobs).
    2. Also request topk_prompt_logprobs: we encode the prompt, generate
       1 token, and read the top-k distribution over the first response
       token via topk_prompt_logprobs on the generation position.

    To use topk_prompt_logprobs for the *generation* position we append
    a dummy token to the prompt so the "last prompt position" lines up
    with the first generation slot.  This gives us top-k (token_id, logprob)
    tuples there.

    Returns dict with:
        "sampled_token": str,
        "sampled_logprob": float | None,
        "choice_logprobs": {choice: logprob},   -- from topk lookup
        "choice_probs": {choice: prob},          -- softmax over choices
        "topk_raw": [(token_id, logprob), ...],  -- full top-k list
    """
    with contextlib.redirect_stdout(io.StringIO()):
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

    # ── Approach 1: generate 1 token, read SampledSequence.logprobs ──
    model_input = types_module.ModelInput.from_ints(prompt_tokens)
    params = types_module.SamplingParams(max_tokens=1, temperature=1.0)
    gen_result = sampling_client.sample(
        prompt=model_input, num_samples=1, sampling_params=params,
    ).result()

    seq = gen_result.sequences[0]
    sampled_token_id = seq.tokens[0] if seq.tokens else None
    sampled_logprob = seq.logprobs[0] if seq.logprobs else None
    sampled_token_str = tokenizer.decode([sampled_token_id]) if sampled_token_id is not None else ""

    # ── Approach 2: topk_prompt_logprobs with dummy appended token ──
    # Append the first choice token as a dummy so the "last prompt position"
    # is the generation slot.  topk there = distribution over first response token.
    dummy_id = choice_token_ids[choices[0]]
    extended_tokens = prompt_tokens + [dummy_id]
    extended_input = types_module.ModelInput.from_ints(extended_tokens)

    topk_result = sampling_client.sample(
        prompt=extended_input,
        num_samples=1,
        sampling_params=types_module.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=topk,
    ).result()

    # topk_prompt_logprobs[-1] = top-k at the last prompt position
    # = distribution over what the model would generate as the first token
    topk_at_gen = topk_result.topk_prompt_logprobs[-1] if topk_result.topk_prompt_logprobs else []

    # Build a lookup: token_id -> logprob from the top-k
    topk_lookup = {tid: lp for tid, lp in topk_at_gen} if topk_at_gen else {}

    # Extract logprobs for our answer choices
    choice_logprobs = {}
    for c in choices:
        tid = choice_token_ids[c]
        if tid in topk_lookup:
            choice_logprobs[c] = topk_lookup[tid]
        else:
            choice_logprobs[c] = None  # not in top-k

    # Normalise to probs (softmax over found choices only)
    found = {c: lp for c, lp in choice_logprobs.items() if lp is not None}
    if found:
        max_lp = max(found.values())
        exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
        total = sum(exps.values())
        choice_probs = {c: exps.get(c, 0.0) / total for c in choices}
    else:
        choice_probs = {c: 0.0 for c in choices}

    return {
        "sampled_token": sampled_token_str.strip(),
        "sampled_logprob": sampled_logprob,
        "choice_logprobs": choice_logprobs,
        "choice_probs": choice_probs,
        "topk_raw": topk_at_gen,
    }


# ── Strategy builders ────────────────────────────────────────────────

def build_end_think_prompt(tokenizer, question, partial_cot: str) -> str:
    """
    Strategy 1: partial CoT → </think> → next token is the answer.
    """
    base = build_thinking_prompt(tokenizer, _user_msg(question), cot_prefix=partial_cot)
    return base + "</think>\n"


def build_cot_anchor_prompt(
    tokenizer, question, partial_cot: str,
    anchor: str = "So, the answer is: ",
) -> str:
    """
    Strategy 2: partial CoT → anchor phrase → </think> → next token is the answer.
    The anchor is injected inside <think>, then we close it.
    """
    cot_with_anchor = partial_cot + " " + anchor if partial_cot else anchor
    base = build_thinking_prompt(tokenizer, _user_msg(question), cot_prefix=cot_with_anchor)
    return base + "</think>\n"


# ── Main comparison loop ─────────────────────────────────────────────

def run_comparison(
    question_id: str,
    rollout_idx: int = 0,
    max_sentences: Optional[int] = 30,
    sentence_stride: int = 1,
    anchors: Optional[List[str]] = None,
    topk: int = TOPK,
) -> Optional[Dict]:
    """Run logprob comparison for one question across its CoT sentences."""
    from tinker import ServiceClient, types
    from transformers import AutoTokenizer

    if anchors is None:
        anchors = COT_ANCHORS[:1]  # default: just "So, the answer is: "

    loaded = load_question_and_cot(question_id, rollout_idx)
    if loaded is None:
        print(f"Could not load question/CoT for {question_id}")
        return None
    question, source_cot = loaded
    choices = get_answer_choices(question)

    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=SUBJECT_MODEL)

    choice_token_ids = _resolve_choice_token_ids(tokenizer, choices)
    print(f"Choice token IDs: {choice_token_ids}")

    cot_segments = get_cumulative_cot_segments(source_cot)
    if max_sentences is not None:
        cot_segments = cot_segments[:max_sentences]

    # Apply stride (keep first + every Nth + last)
    if sentence_stride > 1:
        all_idx = list(range(len(cot_segments)))
        kept = set(all_idx[::sentence_stride])
        kept.add(all_idx[-1])
        indices = sorted(kept)
    else:
        indices = list(range(len(cot_segments)))

    print(f"Question: {question_id}  |  {len(indices)} sentence positions  |  choices: {choices}")
    if isinstance(question, GPQAQuestion):
        print(f"Correct answer: {question.correct_answer}")

    results = []

    for si in tqdm(indices, desc=f"Forcing {question_id}"):
        partial_cot = cot_segments[si]
        row = {"sentence_idx": si, "partial_cot_len": len(partial_cot)}

        # ── Strategy 1: end_think ──
        prompt_et = build_end_think_prompt(tokenizer, question, partial_cot)
        result_et = get_choice_logprobs(
            sampling_client, tokenizer, prompt_et, choices, choice_token_ids, types, topk,
        )
        row["end_think"] = _serialize_result(result_et)

        # ── Strategy 2: cot_anchor(s) ──
        for anchor in anchors:
            anchor_key = anchor.strip().rstrip(": ").replace(" ", "_").lower()
            prompt_ca = build_cot_anchor_prompt(tokenizer, question, partial_cot, anchor)
            result_ca = get_choice_logprobs(
                sampling_client, tokenizer, prompt_ca, choices, choice_token_ids, types, topk,
            )
            row[f"anchor_{anchor_key}"] = {
                "anchor_text": anchor,
                **_serialize_result(result_ca),
            }

        results.append(row)

    # ── Save ──
    output_dir = DATA_DIR / "logprob_comparison" / question_id
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"comparison_{timestamp}.json"

    output = {
        "question_id": question_id,
        "model": SUBJECT_MODEL,
        "choices": choices,
        "choice_token_ids": choice_token_ids,
        "correct_answer": getattr(question, "correct_answer", None),
        "bad_outcome": getattr(question, "bad_outcome", None),
        "num_sentences": len(indices),
        "anchors": anchors,
        "topk": topk,
        "timestamp": datetime.now().isoformat(),
        "sentence_results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    # ── Print summary table ──
    print_summary(output)
    return output


def _serialize_result(result: Dict) -> Dict:
    """Make get_choice_logprobs output JSON-safe (drop topk_raw tuples)."""
    # Store top-10 from topk_raw for inspection without bloating JSON
    topk_decoded = []
    if result.get("topk_raw"):
        # We'd need a tokenizer to decode — just store (token_id, logprob)
        topk_decoded = [(tid, lp) for tid, lp in result["topk_raw"][:10]]
    return {
        "sampled_token": result["sampled_token"],
        "sampled_logprob": result["sampled_logprob"],
        "choice_logprobs": result["choice_logprobs"],
        "choice_probs": result["choice_probs"],
        "top10_tokens": topk_decoded,
    }


def print_summary(output: Dict) -> None:
    """Pretty-print comparison results."""
    choices = output["choices"]
    correct = output.get("correct_answer")
    header = f"{'sent':>4}  {'strategy':<30}  {'sampled':>7}"
    for c in choices:
        mark = "*" if c == correct else " "
        header += f"  {mark}{c:>5}"
    header += "  argmax"

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for row in output["sentence_results"]:
        si = row["sentence_idx"]

        # end_think
        et = row["end_think"]
        probs = et["choice_probs"]
        line = f"{si:4d}  {'end_think':<30}  {et['sampled_token']:>7}"
        for c in choices:
            line += f"  {probs.get(c, 0):6.3f}"
        best = max(probs, key=probs.get) if any(v > 0 for v in probs.values()) else "?"
        line += f"  {best}"
        print(line)

        # anchors
        for key, val in row.items():
            if key.startswith("anchor_"):
                probs = val["choice_probs"]
                label = val["anchor_text"][:28]
                line = f"{'':4s}  {label:<30}  {val['sampled_token']:>7}"
                for c in choices:
                    line += f"  {probs.get(c, 0):6.3f}"
                best = max(probs, key=probs.get) if any(v > 0 for v in probs.values()) else "?"
                line += f"  {best}"
                print(line)
        print()

    print("=" * len(header))
    if correct:
        print(f"(* = correct answer: {correct})")


# ── Entrypoint ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Logprob forcing comparison")
    parser.add_argument("--question_id", type=str, default=None,
                        help="Single question to run (default: all verified)")
    parser.add_argument("--rollout_idx", type=int, default=0)
    parser.add_argument("--max_sentences", type=int, default=30)
    parser.add_argument("--sentence_stride", type=int, default=5)
    parser.add_argument("--topk", type=int, default=TOPK,
                        help="Top-k tokens to request at generation position")
    parser.add_argument("--all_anchors", action="store_true",
                        help="Test all anchor phrases (default: just first)")
    args = parser.parse_args()

    anchors = COT_ANCHORS if args.all_anchors else COT_ANCHORS[:1]

    if args.question_id:
        question_ids = [args.question_id]
    else:
        questions = load_custom_questions(CUSTOM_QUESTIONS_FILE)
        question_ids = [q.id for q in questions if q.id != "blackmail_001"]

    all_outputs = []
    for qid in question_ids:
        out = run_comparison(
            question_id=qid,
            rollout_idx=args.rollout_idx,
            max_sentences=args.max_sentences,
            sentence_stride=args.sentence_stride,
            anchors=anchors,
            topk=args.topk,
        )
        if out:
            all_outputs.append(out)

    if len(all_outputs) > 1:
        print(f"\n{'=' * 60}")
        print(f"Completed {len(all_outputs)} questions.")


if __name__ == "__main__":
    main()
