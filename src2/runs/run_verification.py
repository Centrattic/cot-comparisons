#!/usr/bin/env python3
"""
Run verification rollouts to generate source CoT data.

Produces the verification data that ForcingTask and ResamplingTask
consume via load_question_and_cot().

Usage:
    python -m src2.runs.run_verification
"""

from pathlib import Path

from src2.utils.questions import load_gpqa_questions, load_custom_questions
from src2.utils.verification import run_verification, ensure_verification

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"
VERIFICATION_DIR = DATA_DIR / "verification"

MODEL = "moonshotai/kimi-k2-thinking"
NUM_ROLLOUTS = 50
TEMPERATURE = 0.7
MAX_TOKENS = 8192
MAX_WORKERS = 300
THRESHOLD = 0.8

# Question source: "sample_gpqa", "custom", or "gpqa_hf"
QUESTION_SOURCE = "sample_gpqa"
CUSTOM_QUESTIONS_FILE = None  # Path to questions.json (for "custom" source)
# ──────────────────────────────────────────────────────────────────────


def main():
    if QUESTION_SOURCE == "sample_gpqa":
        questions = load_gpqa_questions(use_samples=True)
    elif QUESTION_SOURCE == "custom":
        questions = load_custom_questions(CUSTOM_QUESTIONS_FILE)
    elif QUESTION_SOURCE == "gpqa_hf":
        from src2.utils.questions import load_gpqa_from_huggingface
        questions = load_gpqa_from_huggingface()
    else:
        raise ValueError(f"Unknown QUESTION_SOURCE: {QUESTION_SOURCE}")

    print(f"Verifying {len(questions)} questions (model={MODEL}, rollouts={NUM_ROLLOUTS})")
    print(f"Data dir: {VERIFICATION_DIR}\n")

    passed = 0
    for q in questions:
        summary = ensure_verification(
            question=q,
            verification_dir=VERIFICATION_DIR,
            num_rollouts=NUM_ROLLOUTS,
            model=MODEL,
            threshold=THRESHOLD,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            max_workers=MAX_WORKERS,
        )
        if summary:
            rate = summary["agreement_rate"]
            status = "PASS" if rate >= THRESHOLD else "FAIL"
            print(f"  {q.id}: agreement={rate:.0%} {status}")
            if rate >= THRESHOLD:
                passed += 1
        else:
            print(f"  {q.id}: below threshold or failed")

    print(f"\n{passed}/{len(questions)} questions passed verification (threshold={THRESHOLD:.0%})")


if __name__ == "__main__":
    main()
