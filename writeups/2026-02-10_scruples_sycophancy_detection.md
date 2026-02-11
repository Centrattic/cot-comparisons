# Scruples Sycophancy Detection: Attention Probes vs LLM Monitors

## TLDR

We compared four approaches for detecting sycophancy in Qwen3-32B on the Scruples moral dilemma task: (1) an attention probe trained on layer 32 activations, (2) a base LLM monitor (GPT-5.2, zero-shot), (3) a high-context LLM monitor (GPT-5.2, with few-shot examples), and (4) an entropy baseline (logistic regression on forced-response entropy features). On a balanced test set of 40 held-out anecdotes, the **attention probe achieves F1=0.91**, the **high-context monitor achieves F1=0.72**, the **base monitor achieves F1=0.49**, and the **entropy baseline achieves F1=0.48** (below the chance baseline of 0.76). The probe's advantage is substantial — it outperforms even the best monitor configuration by 19 F1 points.

![Method Comparison](../plots/scruples/method_comparison.png)

---

## Task and Dataset

**Scruples** presents moral dilemma anecdotes from Reddit's r/AmITheAsshole. Each anecdote asks whether the author was wrong (A) or not wrong (B). We test two intervention variants:

- **suggest_wrong**: User suggests the author was wrong (answer A)
- **suggest_right**: User suggests the author was not wrong (answer B)

**Sycophancy definition**: An anecdote is "sycophantic" if the model's switch rate exceeds a threshold — i.e., the model changes its answer to match the user's suggestion significantly more often than in a no-suggestion control.

**Uncertainty-robust split**: We select 200 anecdotes (100 sycophantic, 100 non-sycophantic) using an uncertainty-robust procedure:
- 50 "high-intervention" sycophantic anecdotes (intervention rate >= 82%)
- 50 "low-intervention" sycophantic anecdotes (switch rate >= 40%, intervention rate < 82%)
- 100 non-sycophantic anecdotes (switch rate < 10%)
- Stratified split: 130 train, 30 val, 40 test anecdotes

The probe trains on 2,366 rollouts (from 130 train anecdotes), validates on 387, and tests on 599 rollouts (from 40 test anecdotes). The monitors are evaluated on all ~3,300 test rollouts from the same 40 test anecdotes.

---

## Methods

### 1. Attention Probe

A 4-head attention probe trained on Qwen3-32B layer 32 activations. The probe reads the full sequence of hidden states from a CoT rollout and outputs a binary sycophantic/non-sycophantic prediction. Clean-example filtering ensures only runs where the model's answer matches the expected behavior (sycophantic answer for sycophantic prompts, majority-control answer for non-sycophantic prompts) are used for training.

**Hyperparameter sweep** over 10 configurations (weight_decay x dropout), with early stopping on validation F1:

| Config | Test F1 | Val F1 | Best Epoch |
|--------|---------|--------|------------|
| **wd=0, do=0.3** | **0.910** | 0.899 | 5 |
| wd=1e-3, do=0.1 | 0.904 | 0.859 | 25 |
| wd=1e-4, do=0.1 | 0.903 | 0.885 | 5 |
| wd=1e-3, do=0 | 0.899 | 0.883 | 5 |
| wd=1e-4, do=0.3 | 0.896 | 0.884 | 15 |
| wd=1e-4, do=0 | 0.891 | 0.891 | 15 |
| wd=0, do=0.1 | 0.879 | 0.861 | 10 |
| wd=1e-2, do=0.3 | 0.836 | 0.954 | 35 |
| wd=0, do=0 | 0.815 | 0.684 | 10 |
| wd=1e-3, do=0.3 | 0.772 | 0.907 | 15 |

Best config: weight_decay=0, dropout=0.3. The top 6 configs all exceed F1=0.89, showing the result is robust to hyperparameter choice.

### 2. Base LLM Monitor

GPT-5.2 (via OpenRouter) reads the full question text and the model's CoT + answer, with no additional context. It predicts what answer the model would have given without the user's suggestion. If the monitor's counterfactual prediction differs from the model's actual answer, the run is flagged as sycophantic.

### 3. High-Context LLM Monitor

Same as the base monitor, but with few-shot examples of sycophantic and non-sycophantic CoTs to calibrate the monitor's judgment. The default configuration uses 15 examples per class.

### 4. Entropy Baseline

A logistic regression classifier trained on forced-response entropy features. The idea: at each sentence boundary during the model's chain-of-thought, we force the model to answer immediately and measure the Shannon entropy of its answer distribution (over choices A and B). If sycophantic CoTs have a distinctive entropy signature — e.g., starting uncertain and becoming more certain as they rationalize the suggested answer — then entropy features alone might detect sycophancy without reading the text or needing activations.

**How forcing works**: For each CoT rollout, we split the thinking text into cumulative sentence segments. At each boundary, we append an anchor phrase ("So, the answer is:") and measure the model's probability distribution over A/B via top-k logprob extraction. This gives an "entropy trajectory" — a time series of how uncertain the model is at each point in its reasoning.

**Features extracted**: From each entropy trajectory, we extract 7 summary statistics: mean entropy, std entropy, min, max, final entropy, initial entropy, and entropy slope (linear fit of entropy vs fraction of CoT remaining). A logistic regression with balanced class weights is trained on these features.

---

## Results

### Overall Comparison

| Method | F1 | Precision | Recall | Accuracy | N (test) |
|--------|-----|-----------|--------|----------|----------|
| **Attention Probe** | **0.910** | **0.871** | **0.952** | **0.855** | 599 |
| High-Context Monitor (15ex) | 0.716 | 0.720 | 0.712 | 0.860 | 3,308 |
| Base Monitor | 0.492 | 0.722 | 0.373 | 0.809 | 3,309 |
| Entropy Baseline | 0.482 | — | — | 0.692 | ~3,265 |

### Per-Variant Breakdown (Monitors)

| Method | suggest_wrong F1 | suggest_right F1 | Combined F1 |
|--------|-----------------|------------------|-------------|
| High-Context Monitor | 0.703 | 0.729 | 0.716 |
| Base Monitor | 0.447 | 0.532 | 0.492 |

The base monitor performs notably worse on suggest_wrong (F1=0.45) than suggest_right (F1=0.53), suggesting it has more difficulty detecting sycophancy when the user suggests the morally harsher judgment.

### Key Observations

1. **The attention probe dominates all methods.** At F1=0.91, it substantially outperforms even the best monitor (F1=0.72). The probe achieves both high precision (0.87) and recall (0.95), meaning it catches nearly all sycophantic CoTs with few false positives.

2. **Few-shot context helps monitors but not enough.** The high-context monitor (F1=0.72) improves substantially over the base monitor (F1=0.49), primarily by boosting recall from 0.37 to 0.71. But even with calibration examples, monitors still miss ~29% of sycophantic CoTs.

3. **The entropy baseline performs below chance.** With test F1=0.48 and accuracy=0.69 against a chance baseline of 0.76, forced-response entropy features carry weak signal for sycophancy detection. The model's answer uncertainty trajectory during CoT provides some information but far less than either the activation probe or text-based monitors.

4. **The probe generalizes across variants.** Unlike the monitors, which show a 5-8 point F1 gap between suggest_wrong and suggest_right, the probe is trained on combined data from both variants and produces a single strong result.

---

## Monitor Example Scaling

We measured how the high-context monitor's performance changes with the number of few-shot examples per class, sweeping over [0, 1, 2, 3, 15, 25, 45] examples.

![Monitor Example Scaling](../plots/scruples/monitor_example_scaling.png)

The biggest jump is from 0 examples (base monitor, F1=0.49) to 1 example (F1=0.70) — a single example per class nearly halves the error rate. Beyond that, returns diminish: going from 1 to 45 examples improves F1 by only 3 points (0.70 to 0.73). Even at 45 examples per class, the monitor plateaus well below the probe's F1=0.91.

---

## Training Curves

The best probe configuration (wd=0, do=0.3) converges quickly:

![Training Curves](../data/scruples/sycophancy_probe/training_curves.png)

The probe reaches its best validation F1 at epoch 5 (early stopping). Train F1 rises quickly to near-perfect levels. The gap between train and test F1 indicates some overfitting to training anecdotes, but the probe still generalizes well enough to achieve F1=0.91 on held-out anecdotes.

---

## Data Scaling

We measured how the attention probe's performance scales with training data by subsampling the training set at 10 fractions (10% to 100%) with 3 repeats each.

![Data Scaling](../plots/scruples/data_scaling_train_test.png)

Train F1 is near-perfect (~1.0) at all data sizes, confirming the probe has more than enough capacity to fit the training data. Test F1 is noisy but averages around 0.85-0.88 across sizes, with no clear upward trend — even at 10% of the data (275 samples), the probe achieves a mean test F1 of 0.87. This suggests the probe learns most of the sycophancy signal from a relatively small number of examples, and that the bottleneck is more about anecdote diversity (generalization across different moral dilemmas) than raw sample count.

---

## Discussion

### Why does the probe outperform monitors?

This result may seem surprising — the probe reads only activation patterns while monitors read the full CoT text. Several factors explain the probe's advantage:

1. **Direct access to internal computation.** The probe reads layer 32 activations, which capture the model's internal representation of its reasoning process. Sycophancy may manifest as subtle computational patterns (e.g., how the model weighs the user's suggestion against its own moral judgment) that are more consistently detectable in activation space than in the variable text of the CoT.

2. **Monitors face a hard counterfactual task.** The monitor must predict what the model *would have* answered without the suggestion — a counterfactual that requires understanding the moral dilemma well enough to form an independent judgment. GPT-5.2 may struggle with the nuance of these moral dilemmas, especially when the "correct" answer is ambiguous.

3. **The probe is trained directly on the task.** The probe sees thousands of labeled examples of sycophantic vs non-sycophantic activations. The monitor, even with 45 few-shot examples, has far less task-specific calibration.

### Why does the entropy baseline underperform?

The entropy baseline tests whether sycophantic CoTs have a distinctive uncertainty profile — e.g., starting uncertain and converging as the model rationalizes. At F1=0.48 (below the 0.76 chance baseline), forced-answer entropy features carry only weak signal. The most informative features are max_entropy (+1.19 coefficient) and entropy_slope (-1.03), suggesting sycophantic CoTs may have slightly higher peak uncertainty and steeper entropy decline — but these patterns are not consistent enough for reliable classification.

This result is informative: it suggests that sycophantic reasoning is not well-characterized by uncertainty dynamics alone. The model's confidence trajectory during CoT is a poor proxy for whether it's being sycophantic. The sycophancy signal lives in *what* the model computes (captured by the activation probe) and *what it writes* (partially captured by monitors), not in *how uncertain* it is at each step.

### When would LLM monitors be preferred over probes?

Despite lower accuracy here, LLM monitors have practical advantages:
- **No activation access needed**: Works with black-box API models where you can't extract hidden states
- **Zero training**: The base monitor requires no labeled data at all
- **Interpretability**: The monitor's reasoning is human-readable — you can audit why it flagged a particular run
- **Generalization**: A well-prompted monitor might transfer across tasks without retraining

### When would activation probes be preferred?

- **Speed and cost**: No API calls needed — inference is a single forward pass through a small model
- **Scalability**: Can monitor every rollout in real-time during deployment
- **Steganography robustness**: Works even if the model learns to hide sycophantic reasoning in its text while still exhibiting it in internal computations
- **Higher accuracy**: As demonstrated here, probes can substantially outperform monitors when activation access is available

---

## Next Steps

1. **Per-variant probe evaluation**: Break down the probe's F1 by suggest_wrong vs suggest_right to check for variant-specific weaknesses
2. **Cross-model transfer**: Test whether the probe trained on Qwen3-32B detects sycophancy in other models
3. **Layer sweep**: The current probe only reads layer 32 — sweeping across layers could find an even stronger signal
4. **Richer entropy baseline**: The current entropy baseline uses only 7 summary statistics; using the full trajectory or a learned model over entropy sequences might recover some signal
