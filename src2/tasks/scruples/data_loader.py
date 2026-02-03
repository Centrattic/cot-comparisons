"""
Data loading and preprocessing for the Scruples task.

This module provides utilities to:
1. Load anecdotes from the Scruples corpus (supports both corpus and anecdotes format)
2. Filter to high-consensus cases (>80% agreement for AUTHOR or OTHER labels)
3. Prepare data in the format expected by the task framework
"""

import json
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd

# Default consensus threshold (80%)
DEFAULT_CONSENSUS_THRESHOLD = 0.80

# Default minimum total votes required
DEFAULT_MIN_TOTAL_VOTES = 50


def load_corpus_jsonl(filepath: Path) -> Iterator[Dict]:
    """
    Load anecdotes from a JSONL file (supports both corpus and anecdotes format).

    Args:
        filepath: Path to the JSONL file

    Yields:
        Dict for each anecdote with keys: id, post_type, title, text, label, label_scores
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_consensus(label_scores: Dict[str, int]) -> tuple[str, float]:
    """
    Compute the consensus ratio for an anecdote's label scores.

    Args:
        label_scores: Dict mapping labels to vote counts

    Returns:
        Tuple of (majority_label, consensus_ratio)
    """
    if not label_scores:
        return "", 0.0

    total_votes = sum(label_scores.values())
    if total_votes == 0:
        return "", 0.0

    majority_label = max(label_scores, key=label_scores.get)
    majority_votes = label_scores[majority_label]
    consensus_ratio = majority_votes / total_votes

    return majority_label, consensus_ratio


def filter_high_consensus(
    anecdotes: Iterator[Dict],
    consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    min_total_votes: int = DEFAULT_MIN_TOTAL_VOTES,
) -> Iterator[Dict]:
    """
    Filter to anecdotes with high consensus for AUTHOR or OTHER labels only.

    Skips anecdotes where:
    - The majority label is EVERYBODY, NOBODY, or INFO
    - The consensus is below the threshold
    - The total number of votes is below min_total_votes

    Args:
        anecdotes: Iterator of anecdote dicts
        consensus_threshold: Minimum consensus ratio required (default 0.80)
        min_total_votes: Minimum total votes required (default 50)

    Yields:
        Anecdotes with high consensus for AUTHOR or OTHER
    """
    for anecdote in anecdotes:
        label_scores = anecdote.get("label_scores", {})
        majority_label, consensus_ratio = compute_consensus(label_scores)
        total_votes = sum(label_scores.values())

        # Only accept AUTHOR or OTHER with high consensus and enough votes
        if (
            majority_label in ("AUTHOR", "OTHER")
            and consensus_ratio >= consensus_threshold
            and total_votes >= min_total_votes
        ):
            # Add consensus info to the anecdote
            anecdote["consensus_ratio"] = consensus_ratio
            anecdote["majority_label"] = majority_label
            anecdote["total_votes"] = total_votes
            yield anecdote


def _find_data_file(data_dir: Path, split: str) -> Path:
    """
    Find the data file for a given split, supporting multiple naming conventions.

    Args:
        data_dir: Directory to search
        split: Which split to load

    Returns:
        Path to the data file

    Raises:
        FileNotFoundError: If no matching file is found
    """
    # Try different naming conventions
    possible_names = [
        f"{split}.scruples-anecdotes.jsonl",  # Real data format
        f"{split}.scruples-corpus.jsonl",  # Test fixtures format
    ]

    for name in possible_names:
        filepath = data_dir / name
        if filepath.exists():
            return filepath

    raise FileNotFoundError(
        f"No data file found for split '{split}' in {data_dir}. Tried: {possible_names}"
    )


def load_scruples_data(
    data_dir: Path,
    split: str = "dev",
    consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Scruples data into a DataFrame, filtered by high consensus.

    Only includes anecdotes where:
    - The majority label is AUTHOR or OTHER (not EVERYBODY, NOBODY, or INFO)
    - The consensus ratio is >= consensus_threshold

    Args:
        data_dir: Directory containing the JSONL files.
        split: Which split to load ("train", "dev", or "test")
        consensus_threshold: Minimum consensus ratio required (default 0.80)
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        DataFrame with columns: id, post_type, title, text, label, label_scores,
                               consensus_ratio, majority_label, author_is_wrong
    """
    data_dir = Path(data_dir)
    filepath = _find_data_file(data_dir, split)

    # Load and filter by consensus
    anecdotes = load_corpus_jsonl(filepath)
    anecdotes = filter_high_consensus(anecdotes, consensus_threshold)

    # Convert to list and apply limit
    anecdotes_list = list(anecdotes)

    if max_samples is not None:
        anecdotes_list = anecdotes_list[:max_samples]

    if not anecdotes_list:
        return pd.DataFrame(
            columns=[
                "id",
                "post_type",
                "title",
                "text",
                "label",
                "label_scores",
                "consensus_ratio",
                "majority_label",
                "author_is_wrong",
            ]
        )

    # Create DataFrame
    df = pd.DataFrame(anecdotes_list)

    # Add author_is_wrong column (True if majority_label is AUTHOR)
    df["author_is_wrong"] = df["majority_label"] == "AUTHOR"

    return df


def get_anecdote_by_id(df: pd.DataFrame, anecdote_id: str) -> Optional[Dict]:
    """
    Get a specific anecdote by ID.

    Args:
        df: DataFrame containing anecdotes
        anecdote_id: The ID to look up

    Returns:
        Dict with anecdote data, or None if not found
    """
    row = df[df["id"] == anecdote_id]
    if len(row) == 0:
        return None
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and filter Scruples data by consensus"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Which split to process",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=DEFAULT_CONSENSUS_THRESHOLD,
        help="Minimum consensus ratio required (default 0.80)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include",
    )

    args = parser.parse_args()

    df = load_scruples_data(
        data_dir=args.data_dir,
        split=args.split,
        consensus_threshold=args.consensus_threshold,
        max_samples=args.max_samples,
    )

    print(f"Loaded {len(df)} high-consensus anecdotes")
    print(f"  - Consensus threshold: {args.consensus_threshold:.0%}")
    print(f"  - Label distribution: {df['majority_label'].value_counts().to_dict()}")
    print(f"  - Author is wrong: {df['author_is_wrong'].sum()} / {len(df)}")

    print("\nSample anecdote:")
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"  ID: {sample['id']}")
        print(f"  Title: {sample['title']}")
        print(f"  Text: {sample['text'][:200]}...")
        print(f"  Label: {sample['label']}")
        print(f"  Consensus: {sample['consensus_ratio']:.1%}")
        print(f"  Author is wrong: {sample['author_is_wrong']}")
