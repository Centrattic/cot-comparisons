from pathlib import Path
from typing import List

from src2.methods import AttentionProbe, ContrastiveSAE, LinearProbe, LlmMonitor
from src2.tasks import ScruplesTask
from src2.tasks.scruples.task import VariantType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

VARIANTS: List[VariantType] = ["first_person", "suggest_wrong", "suggest_right"]
NUM_SAMPLES = 50
MAX_PROMPTS = 300
MAX_WORKERS = 300

v: VariantType

for v in VARIANTS:
    scruples = ScruplesTask(
        subject_model=SUBJECT_MODEL,
        variant=v,
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )

    scruples.run_data(
        num_samples=NUM_SAMPLES,
        max_prompts=MAX_PROMPTS,
    )

    assert scruples.get_data()

    scruples.extract_activations(
        model_name=ACTIVATION_MODEL,
        layers=[LAYER],
        token_positions=["last_thinking", "full_sequence"],
        load_in_4bit=LOAD_IN_4BIT,
    )
