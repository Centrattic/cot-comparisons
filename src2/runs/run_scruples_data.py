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

VARIANTS: List[VariantType] = [
    "suggest_wrong",
    "suggest_right",
    "first_person",
]  # "first_person",
NUM_SAMPLES = 50
ADD_PROMPTS = 150
MAX_WORKERS = 500
LOAD_IN_4BIT = True

v: VariantType
for v in VARIANTS:
    print(v)
    scruples = ScruplesTask(
        subject_model=SUBJECT_MODEL,
        variant=v,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )

    assert scruples.get_data()

    if not scruples.get_data():
        for split in ["dev", "train"]:
            scruples.run_data(
                data_dir=SCRUPLES_DATA_DIR,
                num_samples=NUM_SAMPLES,
                max_prompts=ADD_PROMPTS,
                add=True,
                split=split,
                consensus_threshold=0.0,
            )

    data_slice = scruples.get_sycophancy_slice()

    scruples.extract_activations(
        model_name=ACTIVATION_MODEL,
        layers=[LAYER],
        load_in_4bit=LOAD_IN_4BIT,
        data_slice=data_slice,
    )
