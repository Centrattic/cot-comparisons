"""
Sentence embedding and cosine similarity for CI++ filtering.

Uses bert-large-nli-stsb-mean-tokens (Reimers & Gurevych, 2019) as specified
in the Thought Branches paper for determining semantic similarity between
source sentences and their resampled replacements.
"""

import numpy as np
from typing import List, Optional

EMBEDDING_MODEL = "bert-large-nli-stsb-mean-tokens"

_model_cache = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load and cache the sentence embedding model."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def compute_embeddings(
    sentences: List[str],
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Compute embeddings for a list of sentences.

    Args:
        sentences: List of sentences to embed
        model_name: Sentence transformer model name

    Returns:
        (N, D) numpy array of embeddings
    """
    if not sentences:
        return np.array([])
    model = get_embedding_model(model_name)
    return model.encode(sentences, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_similarity_matrix(
    source_sentence: str,
    resample_continuations: List[List[str]],
    model_name: str = EMBEDDING_MODEL,
) -> List[List[float]]:
    """
    For a source sentence Si, compute cosine similarity of Si to every
    sentence in each resample's continuation.

    Args:
        source_sentence: The original sentence Si being analyzed
        resample_continuations: List of lists -- each inner list is the
            sentences of one resample's continuation (from position i onward)

    Returns:
        List of lists of floats: similarities[j][k] is the cosine similarity
        between Si and the k-th sentence of the j-th resample's continuation
    """
    model = get_embedding_model(model_name)

    # Embed source sentence
    source_emb = model.encode([source_sentence], convert_to_numpy=True)[0]

    # Flatten all resample sentences for batch encoding
    all_sentences = []
    boundaries = [0]
    for continuation in resample_continuations:
        all_sentences.extend(continuation)
        boundaries.append(len(all_sentences))

    if not all_sentences:
        return [[] for _ in resample_continuations]

    # Batch encode all sentences at once
    all_embs = model.encode(all_sentences, convert_to_numpy=True)

    # Reconstruct per-resample similarities
    result = []
    for j in range(len(resample_continuations)):
        start, end = boundaries[j], boundaries[j + 1]
        sims = [
            cosine_similarity(source_emb, all_embs[k])
            for k in range(start, end)
        ]
        result.append(sims)

    return result


def compute_median_threshold(all_similarities: List[List[float]]) -> float:
    """
    Compute the median cosine similarity across all comparisons.

    Per the Thought Branches paper: cosine similarity below median
    determines semantic dissimilarity for CI/CI++ filtering.

    Args:
        all_similarities: similarities[j][k] for each resample j, sentence k

    Returns:
        Median cosine similarity value
    """
    flat = [s for row in all_similarities for s in row]
    if not flat:
        return 0.5
    return float(np.median(flat))
