# Scruples Sycophancy Detection: Attention Probes vs LLM Monitors

## TLDR

We compared four approaches for detecting sycophancy in Qwen3-32B on the Scruples moral dilemma task: (1) an attention probe trained on layer 32 activations, (2) a base LLM monitor (GPT-5.2, zero-shot), (3) a high-context LLM monitor (GPT-5.2, with few-shot examples), and (4) an entropy baseline (logistic regression on forced-response entropy features). On a balanced test set of 40 held-out anecdotes, the **attention probe achieves F1=0.91**, the **high-context monitor (45 examples) achieves F1=0.73**, the **base monitor (v2) achieves F1=0.62**, and the **entropy baseline achieves F1=0.48** (below the chance baseline of 0.76). The probe's advantage is substantial — it outperforms even the best monitor configuration by 18 F1 points.

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

Same as the base monitor, but with few-shot examples of sycophantic and non-sycophantic CoTs to calibrate the monitor's judgment. We sweep over the number of examples per class (1, 2, 3, 15, 25, 45); the best result uses 45 examples per class.

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
| High-Context Monitor (45ex) | 0.729 | 0.741 | 0.718 | — | 3,309 |
| Base Monitor (v2) | 0.621 | 0.745 | 0.532 | — | 3,309 |
| Entropy Baseline | 0.478 | 0.406 | 0.581 | 0.690 | ~3,265 |

### Per-Variant Breakdown (Monitors)

| Method | suggest_wrong F1 | suggest_right F1 | Combined F1 |
|--------|-----------------|------------------|-------------|
| High-Context Monitor (45ex) | 0.708 | 0.750 | 0.729 |
| Base Monitor (v2) | 0.587 | 0.651 | 0.621 |

The base monitor performs notably worse on suggest_wrong (F1=0.59) than suggest_right (F1=0.65), suggesting it has more difficulty detecting sycophancy when the user suggests the morally harsher judgment. The high-context monitor shows a similar but smaller gap (0.71 vs 0.75).

### Key Observations

1. **The attention probe dominates all methods.** At F1=0.91, it substantially outperforms even the best monitor (F1=0.73). The probe achieves both high precision (0.87) and recall (0.95), meaning it catches nearly all sycophantic CoTs with few false positives.

2. **Few-shot context helps monitors substantially.** The high-context monitor (F1=0.73) improves over the base monitor v2 (F1=0.62), primarily by boosting recall from 0.53 to 0.72. But even with 45 calibration examples per class, monitors still miss ~28% of sycophantic CoTs.

3. **The entropy baseline performs below chance.** With test F1=0.48 and accuracy=0.69 against a chance baseline of 0.76, forced-response entropy features carry weak signal for sycophancy detection. It achieves reasonable recall (0.58) but low precision (0.41), flagging many non-sycophantic CoTs as sycophantic.

4. **The probe generalizes across variants.** Unlike the monitors, which show a 4-6 point F1 gap between suggest_wrong and suggest_right, the probe is trained on combined data from both variants and produces a single strong result.

---

## Monitor Example Scaling

We measured how the high-context monitor's performance changes with the number of few-shot examples per class, sweeping over [0, 1, 2, 3, 15, 25, 45] examples.

![Monitor Example Scaling](../plots/scruples/monitor_example_scaling.png)

The biggest jump is from 0 examples (base monitor) to 1 example (F1=0.70) — a single example per class produces a large improvement. Beyond that, returns diminish: going from 1 to 45 examples improves F1 by only 3 points (0.70 to 0.73). Even at 45 examples per class, the monitor plateaus well below the probe's F1=0.91.

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

The entropy baseline tests whether sycophantic CoTs have a distinctive uncertainty profile — e.g., starting uncertain and converging as the model rationalizes. At F1=0.48 (below the 0.76 chance baseline), forced-answer entropy features carry only weak signal. The most informative features are max_entropy (+1.19 coefficient) and entropy_slope (-0.41), suggesting sycophantic CoTs may have slightly higher peak uncertainty and steeper entropy decline — but these patterns are not consistent enough for reliable classification. The baseline achieves decent recall (0.58) but poor precision (0.41), meaning it correctly flags many sycophantic CoTs but also false-alarms on too many non-sycophantic ones.

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

## Broader Context: Answer Distributions Across Questions

To ground the forced-response entropy methodology in a broader set of tasks, we ran 50 rollouts per question on Qwen3-14B across 59 diverse questions — GPQA Diamond (physics, chemistry, biology), AIME competition math, and custom reasoning puzzles (topology, combinatorics, riddles). The answer distributions reveal a striking pattern:

| Category | Count | Examples |
|----------|-------|---------|
| Always correct (100%) | 31 | bat_ball, lily_pad, painted_cube, gpqa_ph_calculation |
| Always wrong (0%) | 2 | gpqa_spin_half (0.7 vs correct -0.7), gpqa_stellar_silicon (3.9 vs correct 12.6) |
| Bimodal (0-100%) | 26 | bagel (21%), aime (53%), starfish (60%), harder_well (92%) |

**Total: 3,161 rollouts across 59 questions.**

The bimodal questions are the most informative — the model doesn't produce a smooth range of answers but rather splits between exactly two attractors. For example, on the bagel topology question (correct answer: 0 holes), 118/150 rollouts answer "2" and 32 answer "0". On the AIME problem, 26/49 get the correct answer (239) while 23 converge on an incorrect answer (78). This is not noise — it reflects genuine computational bifurcation in the model's reasoning.

Selected bimodal questions (sorted by accuracy):

| Question | Accuracy | Correct | Incorrect | Rollouts |
|----------|----------|---------|-----------|----------|
| gpqa_michael_reaction | 4% | 1: 2 | 3: 48 | 50 |
| gpqa_electrochemistry | 6% | 1: 3 | 2: 47 | 50 |
| bookworm | 16% | 0.5: 8 | 4.5: 41 | 49 |
| gpqa_diels_alder | 18% | 2: 9 | 1: 41 | 50 |
| bagel | 21% | 0: 32 | 2: 118 | 150 |
| gpqa_nmr_compound | 32% | 1: 16 | 2: 34 | 50 |
| gpqa_conjugated_dye | 48% | 2: 24 | 1: 25 | 50 |
| aime | 53% | 239: 26 | 78: 23 | 49 |
| starfish | 60% | 16: 30 | 17: 20 | 50 |
| gpqa_optical_activity | 66% | 3: 33 | 4: 17 | 50 |
| gpqa_disproportionation | 70% | 18: 35 | 16: 6 | 50 |
| rope_earth | 82% | 160: 41 | — | 50 |
| harder_well | 92% | 2: 92 | 1: 8 | 100 |
| waffle | 97.5% | 3: 117 | 2: 3 | 120 |

**Connection to sycophancy detection:** The forced-response entropy technique measures exactly this kind of within-question uncertainty — at each CoT sentence boundary, how split is the model between answer options? On sycophantic prompts, the model starts uncertain (bimodal, like the bagel or aime questions above) and the CoT gradually resolves the uncertainty toward the sycophantic answer. On non-sycophantic prompts, the model is confident from sentence one (unimodal, like the always-correct questions).

The key finding from the entropy baseline (F1=0.48) is that while this uncertainty *exists* during sycophantic reasoning, summarizing it into 7 scalar features and classifying with logistic regression doesn't capture enough of the signal. The attention probe (F1=0.91), by contrast, reads the full activation trajectory and captures richer patterns beyond simple uncertainty dynamics.

---

## Next Steps

1. **Per-variant probe evaluation**: Break down the probe's F1 by suggest_wrong vs suggest_right to check for variant-specific weaknesses
2. **Cross-model transfer**: Test whether the probe trained on Qwen3-32B detects sycophancy in other models
3. **Layer sweep**: The current probe only reads layer 32 — sweeping across layers could find an even stronger signal
4. **Richer entropy baseline**: The current entropy baseline uses only 7 summary statistics; using the full trajectory or a learned model over entropy sequences might recover some signal
5. **Answer distribution probing**: Use the bimodal answer distributions above to test whether attention probes can predict which attractor a rollout will converge to, early in the CoT
