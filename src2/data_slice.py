"""
DataSlice — data selection layer for filtering data points across tasks.

Provides optional filtering by IDs, sentence indices, timestamps, and direct
path overrides. None means "no filter" (include all).

Holds optional train/val/test DataFrames (keyed on `filepath` and `label`
columns) for structured split access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Set

import pandas as pd


@dataclass
class DataSlice:
    # instance attributes if type annotations, class atts otherwise
    ids: Optional[Set[str]] = None
    sentence_indices: Optional[Set[int]] = None

    # Timestamp filters
    timestamps: Optional[List[str]] = None
    latest_n: Optional[int] = None

    # Direct path override
    run_paths: Optional[List[Path]] = None

    # Split DataFrames (expected columns: filepath, label, plus task-specific)
    train_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    val_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    test_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    EXPECTED_COLS = ("filepath", "label")

    # ── DataFrame access ─────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Concatenation of all non-None split DataFrames."""
        parts = [x for x in (self.train_df, self.val_df, self.test_df) if x is not None]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    @property
    def train(self) -> DataSlice:
        return DataSlice(train_df=self.train_df)

    @property
    def val(self) -> DataSlice:
        return DataSlice(val_df=self.val_df)

    @property
    def test(self) -> DataSlice:
        return DataSlice(test_df=self.test_df)

    @property
    def filepaths(self) -> List[str]:
        """All filepaths across splits."""
        return list(self.df["filepath"]) if "filepath" in self.df.columns else []

    @property
    def label_series(self) -> pd.Series:
        """All labels across splits."""
        return self.df["label"] if "label" in self.df.columns else pd.Series(dtype=object)

    def labeled(self, label: Any) -> pd.DataFrame:
        """Return rows from df where label matches."""
        return self.df[self.df["label"] == label]

    # ── ID / sentence filtering ──────────────────────────────────

    def matches_id(self, id: str) -> bool:
        return self.ids is None or id in self.ids

    def matches_sentence(self, idx: int) -> bool:
        return self.sentence_indices is None or idx in self.sentence_indices

    def filter_paths(
        self,
        paths: List[Path],
        timestamp_pattern: str = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}|\d{8}_\d{6}",
    ) -> List[Path]:
        """Filter a list of file paths by timestamp directory names.

        If run_paths is set, intersects with those paths.
        If timestamps is set, only includes files under matching timestamp dirs.
        If latest_n is set, sorts timestamps and takes most recent N.
        """
        if self.run_paths is not None:
            run_path_strs = {str(p) for p in self.run_paths}
            paths = [
                p for p in paths if any(str(p).startswith(rp) for rp in run_path_strs)
            ]

        if self.timestamps is not None:
            ts_set = set(self.timestamps)
            paths = [
                p
                for p in paths
                if self._path_has_timestamp(p, ts_set, timestamp_pattern)
            ]

        if self.latest_n is not None:
            all_timestamps = set()
            compiled = re.compile(timestamp_pattern)
            for p in paths:
                for part in p.parts:
                    if compiled.fullmatch(part):
                        all_timestamps.add(part)
            if all_timestamps:
                latest = sorted(all_timestamps, reverse=True)[: self.latest_n]
                latest_set = set(latest)
                paths = [
                    p
                    for p in paths
                    if self._path_has_timestamp(p, latest_set, timestamp_pattern)
                ]

        return paths

    @staticmethod
    def _path_has_timestamp(path: Path, ts_set: Set[str], pattern: str) -> bool:
        for part in path.parts:
            if part in ts_set:
                return True
        return False

    # ── Convenience constructors ─────────────────────────────────

    @classmethod
    def all(cls) -> DataSlice:
        """Select everything (no filters)."""
        return cls()

    @classmethod
    def from_ids(cls, ids) -> DataSlice:
        return cls(ids=set(ids))

    @classmethod
    def latest(cls, n: int = 1) -> DataSlice:
        return cls(latest_n=n)

    @classmethod
    def from_paths(cls, paths: List[Path]) -> DataSlice:
        return cls(run_paths=paths)

    # ── Dunder helpers ───────────────────────────────────────────

    def __len__(self) -> int:
        n = len(self.ids) if self.ids is not None else 0
        df_len = len(self.df)
        return max(n, df_len)

    def __contains__(self, id: object) -> bool:
        return self.matches_id(id)  # type: ignore[arg-type]

    # ── Display ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary."""
        parts: List[str] = []
        n = len(self.ids) if self.ids is not None else "all"
        parts.append(f"DataSlice({n} ids)")
        for name, split_df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
            if split_df is not None:
                parts.append(f"  {name}: {len(split_df)} rows")
        if "label" in self.df.columns:
            counts = self.df["label"].value_counts().to_dict()
            label_strs = [f"{lbl}: {cnt}" for lbl, cnt in sorted(counts.items(), key=lambda x: str(x[0]))]
            parts.append(f"  labels: {', '.join(label_strs)}")
        return "\n".join(parts)
