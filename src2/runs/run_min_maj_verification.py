"""
Generate 250 rollouts per question for the A/B-rephrased test prompts.

These rollouts use A/B answer format (clean single tokens) so they can
be used with the forced-answer entropy baseline.

Questions:
  gpqa_diels_alder_ab   (A = first compound, B = second)
  gpqa_optical_activity_ab  (A = 3, B = 4)

Usage:
    python -m src2.runs.run_min_maj_verification
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src2.utils.questions import load_custom_questions
from src2.utils.verification import ensure_verification

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"
QUESTIONS_FILE = PROJECT_ROOT / "src2" / "utils" / "questions.json"

MODEL = "Qwen/Qwen3-32B"
NUM_ROLLOUTS = 250
TEMPERATURE = 0.7
MAX_TOKENS = 8192
MAX_WORKERS = 300

# All 7 prompts: 5 train + 2 test, with N/M answer labels (no position bias)
TARGET_IDS = {
    # Train
    "bagel_nm",
    "gpqa_nmr_compound_nm",
    "gpqa_benzene_naming_nm",
    "harder_well_nm",
    "bookworm_nm",
    # Test
    "gpqa_diels_alder_nm",
    "gpqa_optical_activity_nm",
}
# ─────────────────────────────────────────────────────────────────────


def main():
    all_questions = load_custom_questions(QUESTIONS_FILE)
    questions = [q for q in all_questions if q.id in TARGET_IDS]

    if not questions:
        raise RuntimeError(
            f"No matching questions found in {QUESTIONS_FILE}. "
            f"Expected IDs: {TARGET_IDS}"
        )

    print(f"Running {NUM_ROLLOUTS} rollouts for {len(questions)} questions "
          f"(model={MODEL})")
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
