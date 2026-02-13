"""
Generate 35 verification rollouts for all questions used in compression experiments.

Questions:
  - 3 custom: starfish, custom_bagel_001, waffle
  - 10 GPQA diamond (selected for high agreement)

Usage:
    python -m src2.runs.run_generate_rollouts
"""

from pathlib import Path

from src2.utils.questions import load_custom_questions, load_gpqa_from_huggingface
from src2.utils.verification import run_verification
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"

MODEL = "Qwen/Qwen3-32B"
NUM_ROLLOUTS = 35
TEMPERATURE = 0.7
MAX_TOKENS = 8192
MAX_WORKERS = 300

# GPQA diamond questions to include (high agreement, model correct)
GPQA_IDS = [
    "gpqa_gpqa_diamond_0007",
    "gpqa_gpqa_diamond_0010",
    "gpqa_gpqa_diamond_0013",
    "gpqa_gpqa_diamond_0019",
    "gpqa_gpqa_diamond_0026",
    "gpqa_gpqa_diamond_0031",
    "gpqa_gpqa_diamond_0034",
    "gpqa_gpqa_diamond_0035",
    "gpqa_gpqa_diamond_0037",
    "gpqa_gpqa_diamond_0039",
]

CUSTOM_IDS = ["starfish", "custom_bagel_001", "waffle"]


def main():
    # Load custom questions
    all_custom = load_custom_questions()
    custom_by_id = {q.id: q for q in all_custom}
    custom_questions = [custom_by_id[qid] for qid in CUSTOM_IDS if qid in custom_by_id]

    # Load GPQA diamond questions from HuggingFace
    gpqa_all = load_gpqa_from_huggingface(subset="gpqa_diamond", max_questions=60)
    gpqa_by_id = {q.id: q for q in gpqa_all}
    gpqa_questions = [gpqa_by_id[qid] for qid in GPQA_IDS if qid in gpqa_by_id]

    questions = custom_questions + gpqa_questions
    print(f"Generating {NUM_ROLLOUTS} rollouts for {len(questions)} questions")
    print(f"  Model: {MODEL}")
    print(f"  Output: {VERIFICATION_DIR}\n")

    for i, q in enumerate(questions):
        # Skip if already has >= NUM_ROLLOUTS rollouts
        qdir = VERIFICATION_DIR / q.id
        if qdir.exists():
            subdirs = sorted([s for s in qdir.iterdir() if s.is_dir()])
            if subdirs:
                latest = subdirs[-1] / "rollouts"
                if latest.exists():
                    n_existing = len(list(latest.glob("rollout_*.json")))
                    if n_existing >= NUM_ROLLOUTS:
                        print(f"[{i+1}/{len(questions)}] {q.id} — already has {n_existing} rollouts, skipping")
                        continue

        print(f"[{i+1}/{len(questions)}] {q.id} (correct={q.correct_answer})")
        summary = run_verification(
            question=q,
            num_rollouts=NUM_ROLLOUTS,
            model=MODEL,
            verification_dir=VERIFICATION_DIR,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            max_workers=MAX_WORKERS,
        )
        rate = summary["agreement_rate"]
        top = summary["most_common_answer"]
        print(f"  -> agreement={rate:.0%} (most common: {top})\n")

    print(f"\nDone. All rollouts saved to {VERIFICATION_DIR}")


if __name__ == "__main__":
    main()
