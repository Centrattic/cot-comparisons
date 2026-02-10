#!/usr/bin/env python3
"""
Logit lens baseline for forced answer entropy prediction.

Projects layer 32 residual stream activations through the model's own
unembedding (final RMSNorm + lm_head) and reads off A/B/C/D probabilities
directly — no training required.

This tests whether the answer distribution is already linearly decodable
at layer 32 via the model's own vocabulary projection.

Usage:
    python -m src2.runs.run_logit_lens_baseline
"""

import json
from pathlib import Path

import numpy as np
import torch

from src2.tasks.forced_response.task import ForcingTask
from src2.utils.activations import ActivationExtractor
from src2.runs.run_entropy_probe import (
    build_question_splits,
    sample_sentence_indices,
    load_probe_data,
    dist_dict_to_entropy,
    compute_metrics,
    print_results,
    ANSWER_LABELS,
    SUBJECT_MODEL,
    ACTIVATION_MODEL,
    DATA_DIR,
    LAYER,
    MAX_SENTENCES_PER_QUESTION_EVAL,
    SEED,
)

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Token IDs for A, B, C, D in Qwen3 tokenizer
CHOICE_TOKEN_IDS = [32, 33, 34, 35]

# Batch size for GPU logit lens computation
LOGIT_LENS_BATCH_SIZE = 256

OUTPUT_PATH = DATA_DIR / "monitor_eval" / "logit_lens_results.json"


# ── Logit lens ────────────────────────────────────────────────────────


def logit_lens_entropy(
    hidden_states: list[np.ndarray],
    extractor: ActivationExtractor,
    batch_size: int = LOGIT_LENS_BATCH_SIZE,
) -> np.ndarray:
    """Apply logit lens to hidden states and compute ABCD entropy.

    For each sample, takes the last-token hidden state, applies the model's
    final norm + lm_head, extracts A/B/C/D logits, softmaxes, and computes
    Shannon entropy.

    Args:
        hidden_states: List of arrays, each [seq_len, hidden_dim].
        extractor: ActivationExtractor with loaded model.
        batch_size: Batch size for GPU computation.

    Returns:
        Array of entropy values, one per sample.
    """
    model = extractor.model
    norm = model.model.norm
    lm_head = model.lm_head

    # Extract last-token hidden state from each sample
    last_token_vecs = []
    for h in hidden_states:
        last_token_vecs.append(h[-1])  # [hidden_dim]

    entropies = []
    choice_ids = torch.tensor(CHOICE_TOKEN_IDS, dtype=torch.long)

    for start in range(0, len(last_token_vecs), batch_size):
        end = min(start + batch_size, len(last_token_vecs))
        batch_np = np.stack(last_token_vecs[start:end])  # [B, hidden_dim]
        batch_t = torch.from_numpy(batch_np).to(
            dtype=norm.weight.dtype, device=norm.weight.device
        )

        with torch.no_grad():
            normed = norm(batch_t)  # [B, hidden_dim]
            logits = lm_head(normed)  # [B, vocab_size]
            abcd_logits = logits[:, choice_ids]  # [B, 4]
            probs = torch.softmax(abcd_logits, dim=-1)  # [B, 4]

        probs_np = probs.cpu().float().numpy()
        for i in range(probs_np.shape[0]):
            dist = {label: float(probs_np[i, j]) for j, label in enumerate(ANSWER_LABELS)}
            ent = dist_dict_to_entropy(dist)
            entropies.append(ent if ent is not None else 0.0)

        if (start // batch_size) % 5 == 0:
            print(f"    Processed {end}/{len(last_token_vecs)} samples...")

    return np.array(entropies, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)

    # ── Step 1: Build question splits (same as entropy probe) ─────────
    print("Building question splits (stratified by entropy)...")
    splits = build_question_splits(forcing.forcing_dir)
    eval_ids = splits["eval_ids"]

    print(f"  Eval questions: {len(eval_ids)}")
    print(f"  GPQA Diamond available on disk: {splits['n_gpqa_available']}")

    # ── Step 2: Sample sentence indices ───────────────────────────────
    print("\nSampling sentence indices...")
    eval_sentence_map = sample_sentence_indices(
        forcing.forcing_dir, eval_ids, MAX_SENTENCES_PER_QUESTION_EVAL, SEED,
    )
    eval_sents = sum(len(v) for v in eval_sentence_map.values())
    print(f"  Eval: {eval_sents} sentence indices ({MAX_SENTENCES_PER_QUESTION_EVAL}/question)")

    # ── Step 3: Load activations (last token from full sequence) ──────
    print("\nLoading eval activations...")
    eval_data = load_probe_data(
        forcing, eval_ids, eval_sentence_map, LAYER, "full_sequence",
    )
    print(f"  Eval: {len(eval_data['X_list'])} samples from {len(eval_ids)} questions")

    if not eval_data["X_list"]:
        print("No eval samples found. Exiting.")
        return

    # ── Step 4: Load model norm + lm_head ─────────────────────────────
    print(f"\nLoading model for logit lens ({ACTIVATION_MODEL})...")
    extractor = ActivationExtractor(model_name=ACTIVATION_MODEL)
    # Trigger model load
    _ = extractor.model
    print("  Model loaded. Using norm + lm_head only (no forward pass).")

    # ── Step 5: Logit lens → entropy ──────────────────────────────────
    print("\nApplying logit lens...")
    pred_entropy = logit_lens_entropy(eval_data["X_list"], extractor)

    # Sanity checks
    max_entropy = np.log(4)
    print(f"\n  Logit lens entropy stats:")
    print(f"    mean={pred_entropy.mean():.4f}, std={pred_entropy.std():.4f}")
    print(f"    min={pred_entropy.min():.4f}, max={pred_entropy.max():.4f}")
    print(f"    (max possible: {max_entropy:.4f})")
    print(f"    in range [0, log(4)]: {np.all((pred_entropy >= 0) & (pred_entropy <= max_entropy + 0.01))}")

    # ── Step 6: Evaluate ──────────────────────────────────────────────
    eval_metrics = compute_metrics(
        eval_data["y_entropy"],
        pred_entropy,
        eval_data["question_ids"],
    )
    print_results(eval_metrics, label="Logit Lens Baseline — Eval Set")

    # ── Step 7: Save results ──────────────────────────────────────────
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output = {
        "method": "logit_lens",
        "layer": LAYER,
        "model": ACTIVATION_MODEL,
        "choice_token_ids": CHOICE_TOKEN_IDS,
        "metrics": eval_metrics,
        "config": {
            "eval_question_ids": eval_ids,
            "max_sentences_per_question": MAX_SENTENCES_PER_QUESTION_EVAL,
            "seed": SEED,
            "n_eval_samples": len(eval_data["X_list"]),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
