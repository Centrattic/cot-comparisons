from pathlib import Path
from typing import List

from src2.data_slice import DataSlice
from src2.methods import AttentionProbe, ContrastiveSAE, LinearProbe, LlmMonitor
from src2.tasks import ScruplesTask
from src2.tasks.scruples.task import VariantType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"
SCRUPLES_DATA_DIR = PROJECT_ROOT / "data" / "scruples_download" / "anecdotes"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

VARIANTS: List[VariantType] = ["suggest_wrong", "suggest_right"]

NUM_SAMPLES = 50
ADD_PROMPTS = 500
MAX_WORKERS = 2000
LOAD_IN_4BIT = True

for v in VARIANTS[:1]:  # suggest_wrong only; change to VARIANTS[1:] for suggest_right
    print(f"\n{'=' * 60}")
    print(f"Starting variant: {v}")
    print(f"{'=' * 60}")
    scruples = ScruplesTask(
        subject_model=SUBJECT_MODEL,
        variant=v,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )

    scruples.run_data(
        data_dir=SCRUPLES_DATA_DIR,
        num_samples=NUM_SAMPLES,
        max_prompts=ADD_PROMPTS,
        split="train",
        consensus_threshold=0.0,
        add=True,
    )

    print(f"Done: {v}")
