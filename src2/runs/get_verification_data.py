from pathlib import Path

from src2.tasks import ForcingTask
from src2.utils.questions import load_custom_questions, load_gpqa_questions
from src2.utils.verification import ensure_verification, run_verification

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"

MODEL = "Qwen/Qwen3-32B"
NUM_ROLLOUTS = 25
TEMPERATURE = 0.7
MAX_TOKENS = 8192
MAX_WORKERS = 250

CUSTOM_QUESTIONS_FILE = (
    Path(__file__).resolve().parent.parent / "utils" / "questions.json"
)

questions = load_custom_questions(CUSTOM_QUESTIONS_FILE)

for q in questions:
    if q.id == "starfish":
        summary = ensure_verification(
            question=q,
            verification_dir=VERIFICATION_DIR,
            num_rollouts=NUM_ROLLOUTS,
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            max_workers=MAX_WORKERS,
        )

        print(summary)
