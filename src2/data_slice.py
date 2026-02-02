"""
DataSlice — thin data selection layer for filtering data points across tasks.

Provides optional filtering by IDs, sentence indices, timestamps, and direct
path overrides. None means "no filter" (include all).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


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

    # Convenience constructors

    @classmethod
    def all(cls) -> "DataSlice":
        """Select everything (no filters)."""
        return cls()

    @classmethod
    def from_ids(cls, ids) -> "DataSlice":
        return cls(ids=set(ids))

    @classmethod
    def latest(cls, n: int = 1) -> "DataSlice":
        return cls(latest_n=n)

    @classmethod
    def from_paths(cls, paths: List[Path]) -> "DataSlice":
        return cls(run_paths=paths)
