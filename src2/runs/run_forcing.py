#!/usr/bin/env python3
"""
Run the Forced Response (forcing) pipeline.

Usage:
    python -m src2.runs.run_forcing
"""

from pathlib import Path

from src2.tasks import ForcingTask
from src2.methods import LlmMonitor, LinearProbe
from src2.tasks.forced_response.prompts import ForcingMonitorPrompt
from src2.utils.questions import load_gpqa_questions
from src2.utils.verification import ensure_verification

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

QUESTION_ID = "gpqa_sample_001"
ROLLOUT_IDX = 0
NUM_FORCES = 5
MAX_SENTENCES = None

# Which steps to run
GENERATE_DATA = True
EXTRACT_ACTIVATIONS = True
RUN_MONITOR = True
RUN_PROBE = True
# ──────────────────────────────────────────────────────────────────────


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # Ensure verification data exists before forcing
    questions = load_gpqa_questions(use_samples=True)
    question = next((q for q in questions if q.id == QUESTION_ID), None)
    if question:
        ensure_verification(
            question=question,
            verification_dir=DATA_DIR / "verification",
            model=SUBJECT_MODEL,
        )

    if GENERATE_DATA:
        forcing.run_data(
            question_id=QUESTION_ID,
            rollout_idx=ROLLOUT_IDX,
            num_forces=NUM_FORCES,
            max_sentences=MAX_SENTENCES,
        )

    if not forcing.get_data():
        print("No forcing data found. Set GENERATE_DATA = True first.")
        return

    if EXTRACT_ACTIVATIONS:
        forcing.extract_activations(
            model_name=ACTIVATION_MODEL,
            layer=LAYER,
            token_position="last_thinking",
        )

    methods = []
    if RUN_MONITOR:
        methods.append(LlmMonitor(prompt=ForcingMonitorPrompt(), model=MONITOR_MODEL))

    if RUN_PROBE:
        methods.append(LinearProbe(layer=LAYER, mode="soft_ce"))

    for m in methods:
        m.set_task(forcing)
        folder = m.get_folder()

        if isinstance(m, LlmMonitor):
            monitor_data = forcing.prepare_for_monitor()
            m.infer(monitor_data)
        elif isinstance(m, LinearProbe):
            probe_data = forcing.get_probe_data(layer=LAYER)
            m.train(probe_data)
            m.infer(probe_data)

        m._output.mark_success()
        print(f"{m.name} results saved to: {folder}")


if __name__ == "__main__":
    main()
