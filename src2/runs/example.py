"""
Example run file showing how to wire tasks and methods together.

All model names, layers, and SAE config are set in the configuration
block at the top — nothing is hardcoded in the task/method classes.
Data lives in PROJECT_ROOT/data/ (outside src2).
method_config.json is auto-saved into each run folder recording
exactly which models/layers/parameters were used.

Usage:
    python -m src2.runs.example
"""

import os
import subprocess
from pathlib import Path

from src2.methods import AttentionProbe, ContrastiveSAE, LinearProbe, LlmMonitor
from src2.tasks.scruples.prompts import ScruplesBaseMonitorPrompt
from src2.tasks import ScruplesTask

# ── Configuration (edit these per-run) ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"

SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

SAE_REPO = "adamkarvonen/qwen3-32b-saes"
SAE_LAYER = 32
SAE_TRAINER = 0

VARIANT = "first_person"
NUM_SAMPLES = 50
MAX_WORKERS = 100
# ──────────────────────────────────────────────────────────────────────

# Initialize task — all config passed in explicitly
scruples = ScruplesTask(
    subject_model=SUBJECT_MODEL,
    variant=VARIANT,
    data_dir=DATA_DIR,
    max_workers=MAX_WORKERS,
)

tasks = [scruples]

for t in tasks:
    if not os.listdir(t.data_dir):
        t.run_data(num_samples=NUM_SAMPLES)

    assert t.get_data()
    assert t.get_activations()

# All methods receive their model/layer/repo explicitly
att = AttentionProbe(layer=LAYER)
probe = LinearProbe(layer=LAYER, mode="ridge")
sae = ContrastiveSAE(sae_repo=SAE_REPO, sae_layer=SAE_LAYER, sae_trainer=SAE_TRAINER)

base_monitor = LlmMonitor(
    prompt=ScruplesBaseMonitorPrompt(VARIANT),
    model=MONITOR_MODEL,
    max_workers=MAX_WORKERS,
)

methods = [att, probe, sae, base_monitor]

for t in tasks:
    for m in methods:
        data = t.get_data(load=True)
        activations = t.get_activations(load=True)

        m.set_task(t)

        # get_folder() auto-saves method_config.json with all
        # task + method attributes (model, layer, SAE repo, etc.)
        folder = m.get_folder()

        with open(f"{folder}/output.txt", "w") as f:
            _ = subprocess.run(["git", "rev-parse", "HEAD"], stdout=f, text=True)
            _ = subprocess.run(["git", "diff"], stdout=f, text=True)

        if isinstance(m, LlmMonitor):
            m.infer(data)
        elif isinstance(m, (ContrastiveSAE, AttentionProbe, LinearProbe)):
            m.train(data, activations)
            m.infer(data)
