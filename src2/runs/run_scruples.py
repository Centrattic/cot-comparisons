#!/usr/bin/env python3
"""
Run the Scruples sycophancy evaluation pipeline.

Usage:
    python -m src2.runs.run_scruples
"""

from pathlib import Path

from src2.tasks import ScruplesTask
from src2.methods import LlmMonitor, LinearProbe, AttentionProbe, ContrastiveSAE
from src2.tasks.scruples.prompts import ScruplesBaseMonitorPrompt, ScruplesDiscriminationPrompt, ScruplesBaselinePrompt

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"

SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

VARIANT = "first_person"
NUM_SAMPLES = 50
MAX_PROMPTS = None
MAX_WORKERS = 100

SAE_REPO = "adamkarvonen/qwen3-32b-saes"
SAE_LAYER = 32
SAE_TRAINER = 0

# Which steps to run
GENERATE_DATA = True
EXTRACT_ACTIVATIONS = True
LOAD_IN_4BIT = False

RUN_MONITOR = True
RUN_BASELINE = True
RUN_DISCRIMINATION = True
RUN_PROBE = True
RUN_ATTENTION_PROBE = True
RUN_SAE = True
# ──────────────────────────────────────────────────────────────────────


def main():
    scruples = ScruplesTask(
        subject_model=SUBJECT_MODEL,
        variant=VARIANT,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )

    if GENERATE_DATA:
        scruples.run_data(
            num_samples=NUM_SAMPLES,
            max_prompts=MAX_PROMPTS,
        )

    if not scruples.get_data():
        print("No data found. Set GENERATE_DATA = True first.")
        return

    if EXTRACT_ACTIVATIONS:
        scruples.extract_activations(
            model_name=ACTIVATION_MODEL,
            layers=[LAYER],
            token_positions=["last_thinking", "full_sequence"],
            load_in_4bit=LOAD_IN_4BIT,
        )

    methods = []

    if RUN_MONITOR:
        methods.append(LlmMonitor(
            prompt=ScruplesBaseMonitorPrompt(VARIANT),
            model=MONITOR_MODEL,
            max_workers=MAX_WORKERS,
        ))

    if RUN_BASELINE:
        methods.append(LlmMonitor(
            prompt=ScruplesBaselinePrompt(),
            model=MONITOR_MODEL,
            max_workers=MAX_WORKERS,
        ))

    if RUN_DISCRIMINATION:
        methods.append(LlmMonitor(
            prompt=ScruplesDiscriminationPrompt(VARIANT),
            model=MONITOR_MODEL,
            max_workers=MAX_WORKERS,
        ))

    if RUN_PROBE:
        methods.append(LinearProbe(layer=LAYER, mode="ridge"))

    if RUN_ATTENTION_PROBE:
        methods.append(AttentionProbe(layer=LAYER))

    if RUN_SAE:
        methods.append(ContrastiveSAE(
            sae_repo=SAE_REPO,
            sae_layer=SAE_LAYER,
            sae_trainer=SAE_TRAINER,
        ))

    for m in methods:
        print(f"\n{'='*60}")
        print(f"Running: {m.name}")
        print(f"{'='*60}")

        m.set_task(scruples)
        folder = m.get_folder()

        if isinstance(m, LlmMonitor):
            monitor_data = scruples.get_monitor_data()
            m.infer(monitor_data)
        elif isinstance(m, (LinearProbe, AttentionProbe, ContrastiveSAE)):
            probe_data = scruples.get_probe_data(layer=LAYER)
            m.train(probe_data)
            m.infer(probe_data)

        m._output.mark_success()
        print(f"Results saved to: {folder}")


if __name__ == "__main__":
    main()
