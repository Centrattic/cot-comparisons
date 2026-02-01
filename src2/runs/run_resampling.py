#!/usr/bin/env python3
"""
Run the Forced Response (resampling) pipeline.

Usage:
    python -m src2.runs.run_resampling
"""

from pathlib import Path

from src2.tasks import ResamplingTask
from src2.methods import LlmMonitor, LinearProbe
from src2.tasks.resampled_response.prompts import ResamplingMonitorPrompt

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

QUESTION_ID = "gpqa_sample_001"
ROLLOUT_IDX = 0
NUM_RESAMPLES = 20
NUM_PREFIX_POINTS = 20

# Which steps to run
GENERATE_DATA = True
EXTRACT_ACTIVATIONS = True
RUN_MONITOR = True
RUN_PROBE = True
# ──────────────────────────────────────────────────────────────────────


def main():
    resampling = ResamplingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    if GENERATE_DATA:
        resampling.run_data(
            question_id=QUESTION_ID,
            rollout_idx=ROLLOUT_IDX,
            num_resamples=NUM_RESAMPLES,
            num_prefix_points=NUM_PREFIX_POINTS,
        )

    if not resampling.get_data():
        print("No resampling data found. Set GENERATE_DATA = True first.")
        return

    if EXTRACT_ACTIVATIONS:
        resampling.extract_activations(
            model_name=ACTIVATION_MODEL,
            layer=LAYER,
            token_position="last_thinking",
        )

    methods = []
    if RUN_MONITOR:
        methods.append(LlmMonitor(prompt=ResamplingMonitorPrompt(), model=MONITOR_MODEL))

    if RUN_PROBE:
        methods.append(LinearProbe(layer=LAYER, mode="soft_ce"))

    for m in methods:
        m.set_task(resampling)
        folder = m.get_folder()

        if isinstance(m, LlmMonitor):
            monitor_data = resampling.prepare_for_monitor()
            m.infer(monitor_data)
        elif isinstance(m, LinearProbe):
            probe_data = resampling.get_probe_data(layer=LAYER)
            m.train(probe_data)
            m.infer(probe_data)

        m._output.mark_success()
        print(f"{m.name} results saved to: {folder}")


if __name__ == "__main__":
    main()
