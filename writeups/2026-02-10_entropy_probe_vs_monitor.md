# Entropy Probe vs LLM Monitor: Predicting Forced Answer Uncertainty from Activations vs Text

## TLDR

We trained a linear probe on Qwen3-32B's internal activations (layer 32) to predict how uncertain the model is at each point in its chain of thought — specifically, the Shannon entropy of the forced answer distribution over A/B/C/D. The probe achieves **R² = 0.40** on held-out questions. A black-box LLM monitor baseline (GPT-5.2 reading the same partial CoT as text) achieves **R² = 0.25** — meaningful but substantially worse. The probe's advantage comes from reading the model's internal state, not just the text it's produced so far.

![Comparison plot](plots/forced_response/entropy_probe_vs_monitor.png)

---

## Why entropy instead of the full distribution?

Our earlier approach tried to predict the full 4-class forced answer distribution (P(A), P(B), P(C), P(D)) from activations. This ran into two problems:

1. **Overfitting to question identity.** With 4 output dimensions per sample, the probe could memorize per-question answer biases rather than learning a general mapping from "how the reasoning is going" to "how certain the model is." Mean-subtraction helped but didn't fully solve it.

2. **Cross-question generalization is the hard part.** The interesting signal isn't "which answer is the model leaning toward" (that's heavily question-dependent and the probe can cheat by recognizing the question), but "how decided is the model right now." Two questions can have completely different answer distributions but the same level of uncertainty — and that's the signal that transfers across questions.

Collapsing the 4-class distribution to a single scalar — Shannon entropy H(p) = -sum(p_i * log(p_i)) — solves both problems. The target is now question-agnostic: entropy ranges from 0 (completely certain) to log(4) = 1.39 (uniform). The probe can't cheat by learning question identity; it has to learn what "uncertain reasoning" looks like in the activation space.

---

## Setup

**Subject model:** Qwen/Qwen3-32B

**Ground truth:** At each sentence boundary in a CoT rollout, we prefill the partial CoT into the model's `<think>` block, append an anchor phrase ("So, the answer is: "), close with `</think>`, and read logprob distributions over answer tokens. The entropy of that distribution is our target.

**Probe:** Mean-pool linear probe on full-sequence activations from layer 32. CoT-only tokens (after `<think>`), with per-question mean subtraction removed. Trained with MSE loss, early stopping on a validation set.

**LLM Monitor baseline:** GPT-5.2 (via OpenRouter) receives the question, choices, and partial CoT as text. It's asked to predict what fraction of independent continuations would arrive at each answer. We parse its predicted distribution and compute entropy.

**Eval split:** 12 held-out questions (10 GPQA Diamond + 2 blackmail scenarios), ~50 sentence boundaries each, stratified by mean entropy to ensure the eval covers the full range.

---

## Results

| Metric | Predict Mean (baseline) | LLM Monitor (GPT-5.2) | Entropy Probe (layer 32) |
|--------|------------------------|----------------------|-------------------------|
| **R²** | 0.000 | 0.250 | **0.397** |
| **MSE** | 0.202 | 0.139 | **0.122** |
| **Pearson r** | 0.000 | 0.505 | **0.654** |
| N samples | 2,970 | 3,737 | 2,970 |

The probe explains ~40% of entropy variance on held-out questions, vs ~25% for the monitor. Both beat the predict-mean baseline substantially.

**What the monitor gets right:** The monitor achieves r = 0.51, meaning it can read partial CoT text and make reasonable predictions about model uncertainty. This makes sense — if the CoT says "I'm not sure about this" or explores multiple approaches, an LLM can pick up on that.

**What the probe adds:** The 15 percentage point R² gap (0.40 vs 0.25) represents information in the activations that isn't expressed in the text. The model "knows" how uncertain it is in ways that go beyond what it's written down. This is consistent with our earlier finding that sycophantic answers carry residual uncertainty in the activation space even when the CoT text sounds confident.

**Monitor failure modes:** The monitor's per-question R² is negative for nearly every question — it can't track within-question entropy variation well. It tends to predict moderate entropy regardless of where we are in the CoT. The monitor's overall R² = 0.25 comes mostly from getting between-question mean entropy approximately right, not from tracking the sentence-by-sentence evolution within a question.

---

## Next steps

1. **Probe on safety-relevant questions.** The current eval is mostly GPQA Diamond (factual science questions). The interesting application is predicting uncertainty on alignment-relevant tasks — can the probe detect when a model is "talking itself into" a harmful answer (high early entropy that decreases, as we saw with sycophancy)?

2. **Better monitor baselines.** The current monitor gets one shot with no examples. A few-shot monitor with examples of "this is what high-entropy vs low-entropy reasoning looks like" could close some of the gap. If it doesn't, that strengthens the case for activation-based monitoring.

3. **Cross-model transfer.** Does a probe trained on Qwen3-32B's activations transfer to other models? If the "what uncertainty looks like in activation space" is model-specific, we need per-model probes. If it transfers, that's a much stronger result.

4. **Entropy as an intervention signal.** If we can detect high-entropy reasoning in real time, we could use it as a trigger for additional oversight — e.g., flag responses where the model's internal uncertainty is high but its output sounds confident.
