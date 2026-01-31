"""
Task class for Principled Resampling (Thought Branches CI++).

Manages data directories and saving results for the CI++ pipeline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class PrincipledResamplingTask:
    """Manages data directories and result storage for CI++ analysis."""

    def __init__(self, model: str = "qwen/qwen3-32b"):
        self.name = "principled_resampling"
        self.model = model
        self.task_dir = Path(__file__).parent.parent.parent.parent / "data" / "principled_resampling"
        self.rollouts_dir = self.task_dir / "rollouts"
        self.ci_dir = self.task_dir / "ci_plus_plus"

        for d in [self.rollouts_dir, self.ci_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def create_rollout_run_dir(self, config: dict) -> Path:
        """Create a timestamped run directory for rollouts with config.json."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.rollouts_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        config["timestamp"] = datetime.now().isoformat()
        config["run_type"] = "rollouts"
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return run_dir

    def save_rollout(self, rollout_idx: int, rollout_data: dict, run_dir: Path) -> Path:
        """Save a single rollout result."""
        rollout_path = run_dir / f"rollout_{rollout_idx:03d}.json"
        with open(rollout_path, "w") as f:
            json.dump(rollout_data, f, indent=2)
        return rollout_path

    def save_rollout_summary(self, rollouts: List[dict], run_dir: Path) -> Path:
        """Save rollout summary with blackmail rate and answer counts."""
        answers = [r.get("answer", "").upper() for r in rollouts if r.get("answer")]
        valid = [a for a in answers if a in ["YES", "NO"]]
        counts = {}
        for a in valid:
            counts[a] = counts.get(a, 0) + 1

        blackmail_count = counts.get("YES", 0)
        total_valid = len(valid)

        summary = {
            "total_rollouts": len(rollouts),
            "valid_rollouts": total_valid,
            "answer_counts": counts,
            "blackmail_count": blackmail_count,
            "blackmail_rate": blackmail_count / total_valid if total_valid > 0 else 0,
            "blackmail_rollout_indices": [
                r["rollout_idx"] for r in rollouts
                if r.get("answer", "").upper() == "YES"
            ],
        }

        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    def create_ci_run_dir(self, rollout_idx: int, config: dict) -> Path:
        """Create a timestamped run directory for CI++ computation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.ci_dir / f"rollout_{rollout_idx:03d}" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        config["timestamp"] = datetime.now().isoformat()
        config["run_type"] = "ci_plus_plus"
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return run_dir

    def save_sentence_resamples(
        self, sentence_idx: int, resamples: List[dict], run_dir: Path
    ) -> Path:
        """Save all resamples for a sentence."""
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}" / "resamples"
        sentence_dir.mkdir(parents=True, exist_ok=True)
        for i, resample in enumerate(resamples):
            path = sentence_dir / f"resample_{i:03d}.json"
            with open(path, "w") as f:
                json.dump(resample, f, indent=2)
        return sentence_dir

    def save_sentence_similarity(
        self, sentence_idx: int, similarity_data: dict, run_dir: Path
    ) -> Path:
        """Save similarity data for a sentence."""
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)
        path = sentence_dir / "similarity.json"
        with open(path, "w") as f:
            json.dump(similarity_data, f, indent=2)
        return path

    def save_sentence_summary(
        self, sentence_idx: int, summary: dict, run_dir: Path
    ) -> Path:
        """Save CI/CI++ summary for a sentence."""
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)
        path = sentence_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def save_ci_results(self, results: dict, run_dir: Path) -> Path:
        """Save the final aggregated CI++ results for all sentences."""
        path = run_dir / "results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path

    def get_latest_rollout_dir(self) -> Optional[Path]:
        """Find the latest timestamped rollout directory."""
        if not self.rollouts_dir.exists():
            return None
        timestamped = sorted(
            [d for d in self.rollouts_dir.iterdir()
             if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
            reverse=True,
        )
        return timestamped[0] if timestamped else None

    def load_rollout(self, rollout_idx: int, run_dir: Optional[Path] = None) -> Optional[dict]:
        """Load a saved rollout by index."""
        if run_dir is None:
            run_dir = self.get_latest_rollout_dir()
        if run_dir is None:
            return None
        path = run_dir / f"rollout_{rollout_idx:03d}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_rollout_summary(self, run_dir: Optional[Path] = None) -> Optional[dict]:
        """Load the rollout summary."""
        if run_dir is None:
            run_dir = self.get_latest_rollout_dir()
        if run_dir is None:
            return None
        path = run_dir / "summary.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def get_latest_ci_run_dir(self, rollout_idx: int) -> Optional[Path]:
        """Find the latest CI++ run directory for a rollout."""
        rollout_dir = self.ci_dir / f"rollout_{rollout_idx:03d}"
        if not rollout_dir.exists():
            return None
        timestamped = sorted(
            [d for d in rollout_dir.iterdir()
             if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
            reverse=True,
        )
        return timestamped[0] if timestamped else None

    def load_ci_results(self, rollout_idx: int, run_dir: Optional[Path] = None) -> Optional[dict]:
        """Load CI++ results for a rollout."""
        if run_dir is None:
            run_dir = self.get_latest_ci_run_dir(rollout_idx)
        if run_dir is None:
            return None
        path = run_dir / "results.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
