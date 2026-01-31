"""
CI and CI++ computation for the Thought Branches method.

Implements:
- CI(Si)   = D_KL[ p(A' | Ti ≉ Si at position i)         || p(A_baseline) ]
- CI++(Si) = D_KL[ p(A' | ∀j>=i: Tj ≉ Si at ALL positions) || p(A_baseline) ]
- Resilience: how many downstream positions contain similar content

Reference: Macar et al. (2025), "Thought Branches: Interpreting LLM
Reasoning Requires Resampling", Equations 1-2.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

LABELS = ("YES", "NO")


def kl_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
    labels: Tuple[str, ...] = LABELS,
    eps: float = 1e-10,
) -> float:
    """
    Compute KL(p || q) for discrete distributions over labels.

    Uses additive smoothing to avoid log(0).

    Args:
        p: Filtered answer distribution (numerator)
        q: Baseline answer distribution (denominator)
        labels: Answer categories
        eps: Smoothing constant

    Returns:
        KL divergence value (non-negative)
    """
    # Normalize with smoothing
    p_vals = [p.get(l, 0.0) + eps for l in labels]
    q_vals = [q.get(l, 0.0) + eps for l in labels]
    p_sum = sum(p_vals)
    q_sum = sum(q_vals)

    kl = 0.0
    for p_v, q_v in zip(p_vals, q_vals):
        p_norm = p_v / p_sum
        q_norm = q_v / q_sum
        kl += p_norm * np.log(p_norm / q_norm)

    return float(max(kl, 0.0))


def compute_answer_distribution(
    answers: List[str],
    labels: Tuple[str, ...] = LABELS,
) -> Dict[str, float]:
    """
    Compute normalized answer distribution from a list of answers.

    Args:
        answers: List of answer strings (e.g., ["YES", "NO", "YES"])
        labels: Valid answer categories

    Returns:
        Dict mapping each label to its proportion
    """
    counts = {l: 0 for l in labels}
    for a in answers:
        a_upper = a.upper().strip()
        if a_upper in counts:
            counts[a_upper] += 1
    total = sum(counts.values())
    if total == 0:
        return {l: 1.0 / len(labels) for l in labels}
    return {l: counts[l] / total for l in labels}


def filter_ci(
    resamples: List[dict],
    all_similarities: List[List[float]],
    threshold: float,
) -> List[dict]:
    """
    CI filter: keep resamples where the first sentence of the continuation
    (position i replacement) is semantically dissimilar to Si.

    A resample passes if its first continuation sentence has cosine
    similarity to Si below the threshold.

    Args:
        resamples: List of resample result dicts
        all_similarities: similarities[j][k] for resample j, sentence k
        threshold: Cosine similarity threshold (below = dissimilar)

    Returns:
        Filtered list of resamples
    """
    filtered = []
    for j, resample in enumerate(resamples):
        if j >= len(all_similarities):
            continue
        sims = all_similarities[j]
        # Check only the first sentence (position i replacement)
        if not sims or sims[0] < threshold:
            filtered.append(resample)
    return filtered


def filter_ci_plus_plus(
    resamples: List[dict],
    all_similarities: List[List[float]],
    threshold: float,
) -> List[dict]:
    """
    CI++ filter: keep resamples where Si's semantic content is absent
    from the ENTIRE downstream trace (all positions j >= i).

    A resample passes if ALL of its continuation sentences have cosine
    similarity to Si below the threshold.

    Args:
        resamples: List of resample result dicts
        all_similarities: similarities[j][k] for resample j, sentence k
        threshold: Cosine similarity threshold (below = dissimilar)

    Returns:
        Filtered list of resamples
    """
    filtered = []
    for j, resample in enumerate(resamples):
        if j >= len(all_similarities):
            continue
        sims = all_similarities[j]
        # Pass if no similar content appears anywhere downstream
        if not sims or all(s < threshold for s in sims):
            filtered.append(resample)
    return filtered


def compute_resilience(
    all_similarities: List[List[float]],
    threshold: float,
) -> List[int]:
    """
    Compute resilience for each resample: the number of downstream
    sentences (from position i onward) where Si's content reappears
    (cosine similarity >= threshold).

    Higher resilience means the content keeps reappearing even after
    removal -- the model tends to regenerate similar reasoning.

    Args:
        all_similarities: similarities[j][k] for resample j, sentence k
        threshold: Cosine similarity threshold

    Returns:
        List of resilience counts (one per resample)
    """
    resiliences = []
    for sims in all_similarities:
        count = sum(1 for s in sims if s >= threshold)
        resiliences.append(count)
    return resiliences


def compute_ci_and_ci_plus_plus(
    resamples: List[dict],
    all_similarities: List[List[float]],
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute CI, CI++, and resilience for one sentence.

    The baseline distribution is computed from ALL resamples (unfiltered).
    CI and CI++ measure how much the answer distribution shifts when we
    filter to resamples where Si's content was successfully removed.

    Args:
        resamples: List of resample result dicts with 'answer' field
        all_similarities: similarities[j][k] for resample j, sentence k
        threshold: Cosine similarity threshold (median)

    Returns:
        Dict with ci, ci_plus_plus, distributions, filter counts, resilience
    """
    # Baseline distribution: all valid resamples
    all_answers = [r["answer"] for r in resamples if r.get("answer")]
    baseline_dist = compute_answer_distribution(all_answers)

    # CI filter and distribution
    ci_filtered = filter_ci(resamples, all_similarities, threshold)
    ci_answers = [r["answer"] for r in ci_filtered if r.get("answer")]
    ci_dist = compute_answer_distribution(ci_answers)
    ci_val = kl_divergence(ci_dist, baseline_dist)

    # CI++ filter and distribution
    cipp_filtered = filter_ci_plus_plus(resamples, all_similarities, threshold)
    cipp_answers = [r["answer"] for r in cipp_filtered if r.get("answer")]
    cipp_dist = compute_answer_distribution(cipp_answers)
    cipp_val = kl_divergence(cipp_dist, baseline_dist)

    # Resilience
    resiliences = compute_resilience(all_similarities, threshold)

    return {
        "ci": ci_val,
        "ci_plus_plus": cipp_val,
        "baseline_distribution": baseline_dist,
        "ci_distribution": ci_dist,
        "ci_plus_plus_distribution": cipp_dist,
        "num_resamples_total": len(resamples),
        "num_valid_answers": len(all_answers),
        "num_filtered_ci": len(ci_filtered),
        "num_filtered_ci_plus_plus": len(cipp_filtered),
        "threshold": threshold,
        "resilience_mean": float(np.mean(resiliences)) if resiliences else 0.0,
        "resilience_max": int(max(resiliences)) if resiliences else 0,
        "p_blackmail_baseline": baseline_dist.get("YES", 0.0),
        "p_blackmail_ci": ci_dist.get("YES", 0.0),
        "p_blackmail_ci_plus_plus": cipp_dist.get("YES", 0.0),
    }
