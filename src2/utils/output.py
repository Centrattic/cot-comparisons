"""
Output management with timestamped folders, latest symlinks, and git logging.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


class OutputManager:
    """
    Manages timestamped output directories for method runs.

    Creates directories like: data/{task}/{method}/YYYY-MM-DD_HH-MM-SS/
    Maintains a `latest` symlink pointing to the most recent successful run.
    Logs git commit hash and diff into each folder.
    """

    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: Parent directory for timestamped run folders.
                      e.g. data/scruples-qwen3-32b/linear_probe_ridge/
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._run_folder: Optional[Path] = None

    def create_run_folder(self) -> Path:
        """
        Create a new timestamped run folder and log git info.

        Returns:
            Path to the created folder, e.g.
            data/{task}/{method}/2026-01-31_14-30-00/
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.base_dir / timestamp
        folder.mkdir(parents=True, exist_ok=True)
        self._run_folder = folder
        self._log_git_info(folder)
        return folder

    def mark_success(self) -> None:
        """
        Update the `latest` symlink to point to the current run folder.

        Should only be called after a run completes successfully.
        """
        if self._run_folder is None:
            raise RuntimeError("No run folder created yet. Call create_run_folder() first.")

        latest = self.base_dir / "latest"

        # Remove existing symlink/file if present; rmdir if it's a real directory
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                import shutil
                shutil.rmtree(latest)
            else:
                latest.unlink()

        # Create relative symlink so it's portable
        latest.symlink_to(self._run_folder.name)

    @property
    def run_folder(self) -> Optional[Path]:
        """The current run folder, or None if not yet created."""
        return self._run_folder

    @staticmethod
    def _log_git_info(folder: Path) -> None:
        """
        Write git rev-parse HEAD and git diff into git_info.txt in the folder.
        Silently skips if not in a git repo.
        """
        git_info_path = folder / "git_info.txt"
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            diff = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True, timeout=30,
            )
            with open(git_info_path, "w") as f:
                f.write(head.stdout)
                if diff.stdout:
                    f.write("\n--- git diff ---\n")
                    f.write(diff.stdout)
        except (subprocess.SubprocessError, FileNotFoundError):
            # Not in a git repo or git not available
            pass
