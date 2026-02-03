"""
Run initial rollouts to generate source CoT data.

Produces the rollout data that ForcingTask and ResamplingTask
consume via load_question_and_cot().

Usage:
    python -m src2.runs.run_verification
"""

from pathlib import Path

from src2.utils.questions import load_custom_questions, load_gpqa_questions
from src2.utils.verification import ensure_verification, run_verification

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"

MODEL = "moonshotai/kimi-k2-thinking"
NUM_ROLLOUTS = 50
TEMPERATURE = 0.7
MAX_WORKERS = 300

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

    print(
        f"Running rollouts for {len(questions)} questions (model={MODEL}, rollouts={NUM_ROLLOUTS})"
    )
    print(f"Data dir: {VERIFICATION_DIR}\n")

    for q in questions:
        summary = ensure_verification(
            question=q,
            verification_dir=VERIFICATION_DIR,
            num_rollouts=NUM_ROLLOUTS,
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            max_workers=MAX_WORKERS,
        )
        rate = summary["agreement_rate"]
        top = summary["most_common_answer"]
        print(f"  {q.id}: agreement={rate:.0%} (most common: {top})")

    print(f"\nDone. Rollouts saved to {VERIFICATION_DIR}")


if __name__ == "__main__":
    main()
