"""
Principled Resampling Task (Thought Branches method)

Implements the Thought Branches resampling method from Macar et al. (2025)
to compute Counterfactual Importance++ (CI++) for each sentence in a CoT rollout.

Paper: "Thought Branches: Interpreting LLM Reasoning Requires Resampling"
       https://arxiv.org/abs/2510.27484

Pipeline:
1. Generate blackmail rollouts with Qwen3-32B
2. For a rollout that exhibits blackmail, split CoT into sentences
3. For each sentence, resample 100 continuations from the prefix before it
4. Compute CI and CI++ using sentence-embedding-based similarity filtering
5. Report which sentences have high causal impact on the blackmail outcome
"""

from .task import PrincipledResamplingTask
from .rollouts import generate_blackmail_rollouts
from .resampling import resample_all_sentences
from .similarity import compute_similarity_matrix, compute_median_threshold
from .metrics import compute_ci_and_ci_plus_plus

__all__ = [
    "PrincipledResamplingTask",
    "generate_blackmail_rollouts",
    "resample_all_sentences",
    "compute_similarity_matrix",
    "compute_median_threshold",
    "compute_ci_and_ci_plus_plus",
]
