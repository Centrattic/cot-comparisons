#!/usr/bin/env python3
"""
Run the Forced Response (forcing) pipeline.

Usage:
    python -m src2.runs.run_forcing
"""

from pathlib import Path
from typing import List

from src2.data_slice import DataSlice
from src2.methods import AttentionProbe, LinearProbe, LlmMonitor
from src2.tasks import ForcingTask
from src2.tasks.forced_response.prompts import ForcingMonitorPrompt
from src2.utils.questions import load_custom_questions, load_gpqa_questions
from src2.utils.verification import ensure_verification

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
MONITOR_MODEL = "openai/gpt-5.2"
ACTIVATION_MODEL = "Qwen/Qwen3-32B"
LAYER = 32

ROLLOUT_IDX = 0
NUM_FORCES = 1
TEMPERATURE = 0.0
MAX_SENTENCES = None
SENTENCE_STRIDE = 5  # only force every Nth sentence (1 = every sentence)

CUSTOM_QUESTIONS_FILE = (
    Path(__file__).resolve().parent.parent / "utils" / "questions.json"
)


forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

questions = load_custom_questions(CUSTOM_QUESTIONS_FILE)
question_map = {q.id: q for q in questions if q.id != "blackmail_001"}
question_ids = question_map.keys()

for qid in question_map.keys():
    question = question_map.get(qid)
    if question:
        ensure_verification(
            question=question,
            verification_dir=PROJECT_ROOT / "data" / "verification_rollouts",
            model=SUBJECT_MODEL,
        )

# generate forcing data
for qid in question_ids:
    forcing.run_data(
        question_id=qid,
        rollout_idx=ROLLOUT_IDX,
        num_forces=NUM_FORCES,
        max_sentences=MAX_SENTENCES,
        temperature=TEMPERATURE,
        sentence_stride=SENTENCE_STRIDE,
    )

assert forcing.get_data()

# extract activations for white box methods
forcing.extract_activations(
    model_name=ACTIVATION_MODEL,
    layer=LAYER,
    data_slice=DataSlice.all(),
)

methods = []
methods.append(LlmMonitor(prompt=ForcingMonitorPrompt(), model=MONITOR_MODEL))

# methods.append(LinearProbe(layer=LAYER, mode="soft_ce"))

for m in methods:
    m.set_task(forcing)

    if isinstance(m, LlmMonitor):
        monitor_data = forcing.prepare_for_monitor(DataSlice.all())
        m.infer(monitor_data)
    elif isinstance(m, LinearProbe):
        probe_data = forcing.get_probe_data(layer=LAYER, data_slice=DataSlice.all())
        m.train(probe_data)
        m.infer(probe_data)

    assert m._output is not None
    m._output.mark_success()

# attention probe
# probe_data = forcing.build_attention_probe_data(
#     layer=LAYER, data_slice=DataSlice.all(), token_position="full_sequence"
# )
# n_samples = len(probe_data["X_list"])
# print(f"\nAttention probe: {n_samples} samples")

# if n_samples < 5:
#     print(f"Skipping attention probe: need >= 5 samples, got {n_samples}")
# else:
#     att_probe = AttentionProbe(layer=LAYER, mode="classification")
#     att_probe.set_task(forcing)
#     att_probe.train(probe_data)
#     att_probe.infer(probe_data)
#     att_probe._output.mark_success()
#     print(f"Attention probe results saved to: {att_probe.get_folder()}")
