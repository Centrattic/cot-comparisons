"""
Data loading and preprocessing for the Scruples task.

This module provides utilities to:
1. Load anecdotes from the Scruples corpus (supports both corpus and anecdotes format)
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

# Path to downloaded anecdotes data
ANECDOTES_DATA_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "scruples_download"
    / "anecdotes"
)

# Add scruples to path for imports
if str(SCRUPLES_REPO_PATH / "src") not in sys.path:
    sys.path.insert(0, str(SCRUPLES_REPO_PATH / "src"))


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
    data_dir: Optional[Path] = None,
    split: str = "dev",
    filter_author_in_wrong: bool = True,
    max_samples: Optional[int] = None,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Load Scruples data into a DataFrame.

    Args:
        data_dir: Directory containing the JSONL files.
                  Defaults to downloaded anecdotes data, falls back to test fixtures.
        split: Which split to load ("train", "dev", or "test")
        filter_author_in_wrong: Whether to filter to only cases where author is wrong
        max_samples: Maximum number of samples to load (None for all)
        offset: Number of samples to skip from the beginning

    Returns:
        DataFrame with columns: id, post_type, title, text, label, label_scores, ground_truth
    """
    if data_dir is None:
        # Try downloaded anecdotes first, then fall back to test fixtures
        if ANECDOTES_DATA_PATH.exists():
            data_dir = ANECDOTES_DATA_PATH
        else:
            data_dir = SCRUPLES_REPO_PATH / "tests" / "fixtures" / "corpus-easy"

    data_dir = Path(data_dir)
    filepath = _find_data_file(data_dir, split)

    # Load and optionally filter
    anecdotes = load_corpus_jsonl(filepath)

    if filter_author_in_wrong:
        anecdotes = filter_author_wrong(anecdotes)

    # Convert to list and apply offset/limit
    anecdotes_list = list(anecdotes)

    # Apply offset first
    if offset > 0:
        anecdotes_list = anecdotes_list[offset:]

    # Then apply max_samples limit
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
                "ground_truth",
            ]
        )

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
    the directory structure for model outputs and monitor results.

    Args:
        data_dir: Source directory containing JSONL files
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

    # Create subdirectories for model outputs and monitor results
    (output_dir / "model" / "control").mkdir(parents=True, exist_ok=True)
    (output_dir / "model" / "intervention").mkdir(parents=True, exist_ok=True)
    (output_dir / "monitor" / "control").mkdir(parents=True, exist_ok=True)
    (output_dir / "monitor" / "intervention").mkdir(parents=True, exist_ok=True)

    # Add path columns (these will be populated when model outputs are collected)
    df["model_control_path"] = df["id"].apply(lambda x: f"model/control/{x}.json")
    df["model_intervention_path"] = df["id"].apply(
        lambda x: f"model/intervention/{x}.json"
    )
    df["monitor_control_path"] = df["id"].apply(lambda x: f"monitor/control/{x}.json")
    df["monitor_intervention_path"] = df["id"].apply(
        lambda x: f"monitor/intervention/{x}.json"
    )

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

    parser = argparse.ArgumentParser(
        description="Prepare Scruples data for the task framework"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing JSONL files",
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
