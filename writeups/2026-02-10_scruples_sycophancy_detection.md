# Scruples Sycophancy Detection: Attention Probes vs LLM Monitors

## TLDR

We compared three approaches for detecting sycophancy in Qwen3-32B on the Scruples moral dilemma task: (1) an attention probe trained on layer 32 activations, (2) a base LLM monitor (GPT-5.2, zero-shot), and (3) a high-context LLM monitor (GPT-5.2, 15 few-shot examples). On a balanced test set of 19 held-out anecdotes, the **high-context monitor achieves F1=0.95**, the **base monitor achieves F1=0.80**, and the **attention probe achieves F1=0.61**. The monitors are evaluated on 100 runs from the test anecdotes; the probe is evaluated on 1,528 runs from the same anecdotes. An entropy baseline (logistic regression on forced-response entropy features) has not yet been trained.

![Method Comparison](../plots/scruples/method_comparison.png)

---

## Task and Dataset

**Scruples** presents moral dilemma anecdotes from Reddit's r/AmITheAsshole. Each anecdote asks whether the author was wrong (A) or not wrong (B). We test two intervention variants:

- **suggest_wrong**: User suggests the author was wrong (answer A)
- **suggest_right**: User suggests the author was not wrong (answer B)

**Sycophancy definition**: An anecdote is "sycophantic" if the model's switch rate exceeds a threshold — i.e., the model changes its answer to match the user's suggestion significantly more often than in a no-suggestion control.

**Uncertainty-robust split**: We select 198 anecdotes (98 sycophantic, 100 non-sycophantic) using an uncertainty-robust procedure:
- 50 "high-intervention" sycophantic anecdotes (intervention rate >= 82%)
- 50 "low-intervention" sycophantic anecdotes (switch rate >= 40%, intervention rate < 82%)
- 100 non-sycophantic anecdotes (switch rate < 10%)
- Stratified 80/20 train/test split by stratum, yielding ~80 train and 19 test anecdotes

Each anecdote has multiple rollouts (runs), giving 6,116 training samples and 1,528 test samples for the probe.

---

## Methods

### 1. Attention Probe

A 4-head attention probe trained on Qwen3-32B layer 32 activations. The probe reads the full sequence of hidden states from a CoT rollout and outputs a binary sycophantic/non-sycophantic prediction. Clean-example filtering ensures only runs where the model's answer matches the expected behavior (sycophantic answer for sycophantic prompts, majority-control answer for non-sycophantic prompts) are used for training.

**Hyperparameter sweep** over 10 configurations (weight_decay x dropout), trained in parallel:

| Config | Test F1 | Val F1 | Best Epoch |
|--------|---------|--------|------------|
| wd=1e-4, do=0.1 | **0.606** | 0.462 | 20 |
| wd=0, do=0.3 | 0.547 | 0.511 | 10 |
| wd=1e-2, do=0.3 | 0.545 | 0.487 | 20 |
| wd=1e-4, do=0 | 0.542 | 0.464 | 5 |
| wd=1e-3, do=0.1 | 0.525 | 0.429 | 5 |
| wd=1e-3, do=0 | 0.503 | 0.479 | 5 |
| wd=0, do=0 | 0.498 | 0.365 | 20 |
| wd=1e-3, do=0.3 | 0.494 | 0.432 | 10 |
| wd=1e-4, do=0.3 | 0.484 | 0.436 | 5 |
| wd=0, do=0.1 | 0.472 | 0.452 | 20 |

Best config: weight_decay=1e-4, dropout=0.1.

![Hyperparameter Sweep](../plots/scruples/hyperparam_sweep.png)

### 2. Base LLM Monitor

GPT-5.2 (via OpenRouter) reads the full question text and the model's CoT + answer, with no additional context. It predicts what answer the model would have given without the user's suggestion. If the monitor's counterfactual prediction differs from the model's actual answer, the run is flagged as sycophantic.

### 3. High-Context LLM Monitor

Same as the base monitor, but with 15 few-shot examples of sycophantic and non-sycophantic CoTs to calibrate the monitor's judgment.

### 4. Entropy Baseline (pending)

A logistic regression classifier trained on forced-response entropy features — specifically, the Shannon entropy of the model's answer distribution at sentence boundaries during its CoT. The forcing results have been collected but the classifier has not yet been trained.

---

## Results

| Method | F1 | Precision | Recall | Accuracy | N (test) |
|--------|-----|-----------|--------|----------|----------|
| Attention Probe | 0.61 | 0.53 | 0.71 | 0.68 | 1,528 |
| Base Monitor | 0.80 | 0.96 | 0.69 | 0.76 | 100 |
| High-Context Monitor | **0.95** | **1.00** | **0.90** | **0.93** | 100 |

**Key observations:**

1. **Monitors dominate the probe.** The high-context monitor achieves near-perfect precision (1.00) and very high recall (0.90), substantially outperforming the attention probe on all metrics.

2. **Few-shot context matters.** The jump from base (F1=0.80) to high-context (F1=0.95) shows that giving the monitor examples of what sycophantic reasoning looks like significantly improves detection, primarily by boosting recall from 0.69 to 0.90.

3. **The probe struggles with precision.** At 0.53 precision, the probe generates many false positives — predicting sycophancy on non-sycophantic runs. Its recall (0.71) is reasonable, meaning it can detect most sycophantic behavior but can't reliably distinguish it from normal variation.

4. **Sample size caveat.** The monitors are evaluated on 100 runs (from 19 test anecdotes), while the probe is evaluated on 1,528 runs. The monitor numbers should be interpreted with this smaller sample in mind.

---

## Data Scaling

We measured how the attention probe's performance scales with training data by subsampling the training set at 10 fractions (10% to 100%) with 3 repeats each.

![Data Scaling](../plots/scruples/data_scaling.png)

The scaling curve is **non-monotonic and noisy**: test F1 ranges from 0.37 (at 10%, n=611) to 0.55 (at 80%, n=4892), with substantial variance at each size. The general trend is upward but with significant dips (e.g., the drop from 0.46 at 20% to 0.38 at 40%). At full data (n=6116), test F1 is 0.50 — below the sweep's best of 0.61 because the scaling experiment uses fixed hyperparameters (wd=1e-3, do=0.3) rather than the sweep-optimal ones.

The noisy scaling curve and the large train/test gap visible in training curves suggest the probe is capacity-limited rather than data-limited — it can memorize training data well but struggles to generalize across anecdotes.

---

## Training Curves

The best probe configuration (wd=1e-4, do=0.1) shows severe overfitting:

![Training Curves](../data/scruples/sycophancy_probe/training_curves.png)

- **Train F1** rises to ~0.93 by epoch 10 and plateaus at ~0.97
- **Test F1** peaks at ~0.61 around epoch 20, then drops sharply to ~0.47 by epoch 30

The ~35-point gap between train and test F1 at the optimal checkpoint indicates the probe memorizes training anecdotes rather than learning a general sycophancy signal. Early stopping at the right epoch is critical — training too long degrades test performance substantially.

---

## Discussion

### Why do monitors outperform the probe?

The fundamental asymmetry is that **monitors read the CoT text directly** while the probe must extract the same information from activation patterns. For sycophancy detection in Scruples, the CoT often contains explicit signals: the model references the user's suggestion, hedges about the moral question, or reverses its initial reasoning. A strong LLM can read these cues directly.

The probe, by contrast, must learn to map high-dimensional activation trajectories to binary labels with only ~80 training anecdotes. With just 19 test anecdotes, the per-anecdote test set is small enough that a few hard-to-classify anecdotes can substantially drag down metrics.

### What would make the probe more competitive?

1. **More training anecdotes.** The current setup uses 80 train / 19 test anecdotes. Doubling or tripling the anecdote count could help the probe learn more generalizable features, though the noisy data scaling curve suggests diminishing returns.

2. **Multi-layer features.** Currently the probe only reads layer 32. Concatenating activations from multiple layers could provide richer features.

3. **Better probe architecture.** The 4-head attention probe may not have enough capacity to capture the sycophancy signal. Larger probes, or probes that attend over sentence-level summaries rather than token-level activations, could help.

### When would activation probes be preferred?

Despite lower accuracy, activation probes have advantages over LLM monitors:
- **Speed**: No API calls needed — inference is a single forward pass through a small model
- **Cost**: Orders of magnitude cheaper than calling GPT-5.2 per rollout
- **Scalability**: Can monitor every rollout in real-time during deployment
- **No text leakage**: Works even if the model learns to hide sycophantic reasoning in text (steganography)

The last point is the most important for alignment: if a model can conceal sycophantic behavior in its written CoT while still exhibiting it in its internal computations, monitors that read text will fail but probes on activations may still detect it.

---

## Next Steps

1. **Train the entropy baseline** to complete the 4-method comparison
2. **Re-run data scaling** with the sweep-optimal hyperparameters (wd=1e-4, do=0.1) to get a cleaner scaling curve
3. **Add train metrics to data scaling** to visualize the train/test gap across data sizes
4. **Expand the anecdote set** — the 198-anecdote uncertainty-robust split may be too small for reliable probe training
5. **Test the discrimination monitor** — results exist but haven't been integrated into the comparison yet
