"""
Data loading and preprocessing for the Scruples task.

This module provides utilities to:
1. Load anecdotes from the Scruples corpus
2. Filter to cases where the author is in the wrong (AUTHOR or EVERYBODY labels)
3. Prepare data in the format expected by the task framework
"""

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

# Path to the cloned scruples repository
SCRUPLES_REPO_PATH = Path(__file__).parent.parent.parent.parent / "scruples"

# Add scruples to path for imports
if str(SCRUPLES_REPO_PATH / "src") not in sys.path:
    sys.path.insert(0, str(SCRUPLES_REPO_PATH / "src"))


def load_corpus_jsonl(filepath: Path) -> Iterator[Dict]:
    """
    Load anecdotes from a JSONL corpus file.

    Args:
        filepath: Path to the .scruples-corpus.jsonl file

    Yields:
        Dict for each anecdote with keys: id, post_type, title, text, label, label_scores
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def filter_author_wrong(anecdotes: Iterator[Dict]) -> Iterator[Dict]:
    """
    Filter to anecdotes where the community voted the author is in the wrong.

    The author is considered "in the wrong" if the label is:
    - AUTHOR (YTA - You're the asshole)
    - EVERYBODY (ESH - Everyone sucks here)

    Args:
        anecdotes: Iterator of anecdote dicts

    Yields:
        Anecdotes where the author is in the wrong
    """
    for anecdote in anecdotes:
        label = anecdote.get("label", "")
        if label in ("AUTHOR", "EVERYBODY"):
            yield anecdote


def load_scruples_data(
    data_dir: Optional[Path] = None,
    split: str = "dev",
    filter_author_in_wrong: bool = True,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Scruples corpus data into a DataFrame.

    Args:
        data_dir: Directory containing the corpus JSONL files.
                  Defaults to scruples/tests/fixtures/corpus-easy/ for testing.
        split: Which split to load ("train", "dev", or "test")
        filter_author_in_wrong: Whether to filter to only cases where author is wrong
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        DataFrame with columns: id, post_type, title, text, label, label_scores, ground_truth
    """
    if data_dir is None:
        # Default to test fixtures for development
        data_dir = SCRUPLES_REPO_PATH / "tests" / "fixtures" / "corpus-easy"

    data_dir = Path(data_dir)
    filepath = data_dir / f"{split}.scruples-corpus.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    # Load and optionally filter
    anecdotes = load_corpus_jsonl(filepath)

    if filter_author_in_wrong:
        anecdotes = filter_author_wrong(anecdotes)

    # Convert to list and limit if needed
    anecdotes_list = list(anecdotes)

    if max_samples is not None:
        anecdotes_list = anecdotes_list[:max_samples]

    if not anecdotes_list:
        return pd.DataFrame(columns=["id", "post_type", "title", "text", "label", "label_scores", "ground_truth"])

    # Create DataFrame
    df = pd.DataFrame(anecdotes_list)

    # Add ground_truth column (1 = author is in the wrong, which is always true after filtering)
    df["ground_truth"] = 1

    return df


def prepare_task_data(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prepare Scruples data in the format expected by the task framework.

    This creates a data.csv file with the required columns and sets up
    the directory structure for CoT/response files.

    Args:
        data_dir: Source directory containing corpus JSONL files
        output_dir: Output directory for processed data.
                    Defaults to data/scruples/ in the project root.
        split: Which split to process
        max_samples: Maximum number of samples to include

    Returns:
        The prepared DataFrame
    """
    # Load filtered data
    df = load_scruples_data(
        data_dir=data_dir,
        split=split,
        filter_author_in_wrong=True,
        max_samples=max_samples,
    )

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "scruples"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for CoT and responses
    (output_dir / "cots").mkdir(exist_ok=True)
    (output_dir / "responses").mkdir(exist_ok=True)

    # Add path columns (these will be populated when model outputs are collected)
    df["cot_path"] = df["id"].apply(lambda x: f"cots/{x}.txt")
    df["response_path"] = df["id"].apply(lambda x: f"responses/{x}.txt")

    # Save data.csv
    df.to_csv(output_dir / "data.csv", index=False)

    print(f"Prepared {len(df)} anecdotes in {output_dir}")
    print(f"  - Label distribution: {df['label'].value_counts().to_dict()}")

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

    parser = argparse.ArgumentParser(description="Prepare Scruples data for the task framework")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing corpus JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Which split to process",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include",
    )

    args = parser.parse_args()

    df = prepare_task_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
    )

    print("\nSample anecdote:")
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"  ID: {sample['id']}")
        print(f"  Title: {sample['title']}")
        print(f"  Text: {sample['text'][:200]}...")
        print(f"  Label: {sample['label']}")
