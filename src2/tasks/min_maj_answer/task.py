"""
Majority/minority answer classification task.

Given a prompt and a single rollout, classify whether the rollout's answer
is the majority or minority answer among all rollouts for that prompt.

Key constraint: methods cannot access other rollouts on the SAME prompt —
only rollouts from OTHER prompts (e.g. for few-shot examples).

Data source: /home/riya/neel-projs/global-cot-analysis/prompts/
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ...data_slice import DataSlice
from ..base import BaseTask

ROLLOUTS_ROOT = Path("/home/riya/neel-projs/global-cot-analysis/prompts")

ALL_PROMPT_IDS = [
    "bagel",
    "gpqa_nmr_compound",
    "gpqa_benzene_naming",
    "harder_well",
    "bookworm",
    "gpqa_diels_alder",
    "gpqa_optical_activity",
]


class MinMajAnswerTask(BaseTask):
    """
    Task: classify a single rollout as majority or minority answer.

    Loads rollouts from the global-cot-analysis prompts directory,
    computes per-prompt majority/minority labels, and serves individual
    rollouts to methods for classification.

    Args:
        prompt_ids: which prompt categories to load (e.g. ["gpqa_diels_alder"])
        model: which model's rollouts to use (default: "qwen3-32b")
        rollouts_root: path to the global-cot-analysis prompts directory
        data_dir: output directory for this task's data
    """

    def __init__(
        self,
        prompt_ids: List[str],
        model: str = "qwen3-32b",
        rollouts_root: Path = ROLLOUTS_ROOT,
        data_dir: Optional[Path] = None,
    ):
        super().__init__(name="min_maj_answer", data_dir=data_dir)
        self.prompt_ids = prompt_ids
        self.model = model
        self.rollouts_root = Path(rollouts_root)
        self._prompts_json = None

    @classmethod
    def loo_folds(
        cls, prompt_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return leave-one-out fold definitions.

        Each fold holds out one prompt as test and uses the rest as train.

        Returns:
            List of dicts with keys: fold_idx, test_prompt_id, train_prompt_ids
        """
        if prompt_ids is None:
            prompt_ids = ALL_PROMPT_IDS
        folds = []
        for i, test_pid in enumerate(prompt_ids):
            folds.append({
                "fold_idx": i,
                "test_prompt_id": test_pid,
                "train_prompt_ids": [p for p in prompt_ids if p != test_pid],
            })
        return folds

    def _load_prompts_json(self) -> Dict[str, str]:
        if self._prompts_json is None:
            with open(self.rollouts_root / "prompts.json") as f:
                self._prompts_json = json.load(f)
        return self._prompts_json

    def _load_rollouts_for_prompt(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Load all rollout JSON files for a given prompt + model."""
        rollouts_dir = self.rollouts_root / prompt_id / self.model / "rollouts"
        if not rollouts_dir.exists():
            return []

        rollouts = []
        for f in sorted(rollouts_dir.iterdir()):
            if f.suffix == ".json":
                with open(f) as fh:
                    data = json.load(fh)
                    data["rollout_file"] = str(f)
                    data["rollout_idx"] = int(f.stem)
                    rollouts.append(data)
        return rollouts

    def _compute_labels(self, rollouts: List[Dict]) -> List[Dict]:
        """Add majority/minority labels based on answer distribution."""
        # Filter out rollouts with empty answers
        rollouts = [
            r for r in rollouts
            if r.get("processed_response_content", r.get("response_content", "")).strip()
        ]
        answers = [
            r.get("processed_response_content", r.get("response_content", ""))
            for r in rollouts
        ]
        counts = Counter(answers)
        if not counts:
            return rollouts

        majority_answer = counts.most_common(1)[0][0]
        total = sum(counts.values())
        majority_count = counts[majority_answer]
        majority_frac = majority_count / total

        for r, ans in zip(rollouts, answers):
            r["answer"] = ans
            r["is_majority"] = ans == majority_answer
            r["label"] = "majority" if ans == majority_answer else "minority"
            r["majority_answer"] = majority_answer
            r["majority_frac"] = majority_frac
            r["answer_counts"] = dict(counts)

        return rollouts

    def _build_rollout_df(self, prompt_ids: List[str]) -> pd.DataFrame:
        """Build a DataFrame of labeled rollouts for the given prompt IDs."""
        prompts_json = self._load_prompts_json()
        rows = []
        for pid in prompt_ids:
            rollouts = self._load_rollouts_for_prompt(pid)
            if not rollouts:
                continue
            rollouts = self._compute_labels(rollouts)
            prompt_text = prompts_json.get(pid, "")
            for r in rollouts:
                rows.append({
                    "prompt_id": pid,
                    "prompt_text": prompt_text,
                    "rollout_idx": r["rollout_idx"],
                    "cot_content": r.get("cot_content", ""),
                    "response_content": r.get("response_content", ""),
                    "answer": r["answer"],
                    "label": r["label"],
                    "is_majority": r["is_majority"],
                    "majority_answer": r["majority_answer"],
                    "majority_frac": r["majority_frac"],
                    "answer_counts": json.dumps(r["answer_counts"]),
                    "filepath": r.get("rollout_file", ""),
                })
        return pd.DataFrame(rows)

    def run_data(self, **kwargs) -> None:
        """Load rollouts, compute labels, and save to data_dir."""
        df = self._build_rollout_df(self.prompt_ids)
        if df.empty:
            print("Warning: no rollouts found for any prompt")
            return
        df.to_csv(self.data_dir / "rollouts.csv", index=False)
        print(f"Saved {len(df)} rollouts to {self.data_dir / 'rollouts.csv'}")

        # Also save per-prompt summary
        summary_rows = []
        for prompt_id in self.prompt_ids:
            subset = df[df["prompt_id"] == prompt_id]
            if subset.empty:
                continue
            summary_rows.append({
                "prompt_id": prompt_id,
                "n_rollouts": len(subset),
                "n_majority": int(subset["is_majority"].sum()),
                "n_minority": int((~subset["is_majority"]).sum()),
                "majority_frac": float(subset["majority_frac"].iloc[0]),
                "majority_answer": subset["majority_answer"].iloc[0],
                "answer_counts": subset["answer_counts"].iloc[0],
            })
        pd.DataFrame(summary_rows).to_csv(
            self.data_dir / "prompt_summary.csv", index=False
        )

    def get_data(self, load: bool = False) -> Union[bool, Optional[Dict[str, pd.DataFrame]]]:
        csv_path = self.data_dir / "rollouts.csv"
        if not load:
            return csv_path.exists()
        if not csv_path.exists():
            return None
        return {
            "rollouts": pd.read_csv(csv_path),
            "summary": pd.read_csv(self.data_dir / "prompt_summary.csv"),
        }

    def get_activations(self, load: bool = False) -> Union[bool, Optional[Any]]:
        # No activations for black-box monitor method
        return False if not load else None

    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Evaluate majority/minority predictions."""
        if not predictions or not ground_truth:
            return {"accuracy": 0.0}

        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        total = len(predictions)

        # Compute per-class metrics
        tp_maj = sum(1 for p, g in zip(predictions, ground_truth) if p == "majority" and g == "majority")
        fp_maj = sum(1 for p, g in zip(predictions, ground_truth) if p == "majority" and g != "majority")
        fn_maj = sum(1 for p, g in zip(predictions, ground_truth) if p != "majority" and g == "majority")

        tp_min = sum(1 for p, g in zip(predictions, ground_truth) if p == "minority" and g == "minority")
        fp_min = sum(1 for p, g in zip(predictions, ground_truth) if p == "minority" and g != "minority")
        fn_min = sum(1 for p, g in zip(predictions, ground_truth) if p != "minority" and g == "minority")

        prec_maj = tp_maj / (tp_maj + fp_maj) if (tp_maj + fp_maj) > 0 else 0.0
        rec_maj = tp_maj / (tp_maj + fn_maj) if (tp_maj + fn_maj) > 0 else 0.0
        prec_min = tp_min / (tp_min + fp_min) if (tp_min + fp_min) > 0 else 0.0
        rec_min = tp_min / (tp_min + fn_min) if (tp_min + fn_min) > 0 else 0.0

        f1_maj = 2 * prec_maj * rec_maj / (prec_maj + rec_maj) if (prec_maj + rec_maj) > 0 else 0.0
        f1_min = 2 * prec_min * rec_min / (prec_min + rec_min) if (prec_min + rec_min) > 0 else 0.0
        macro_f1 = (f1_maj + f1_min) / 2

        return {
            "accuracy": correct / total,
            "macro_f1": macro_f1,
            "majority_f1": f1_maj,
            "minority_f1": f1_min,
            "majority_precision": prec_maj,
            "majority_recall": rec_maj,
            "minority_precision": prec_min,
            "minority_recall": rec_min,
            "n_total": total,
            "n_correct": correct,
        }

    def get_train_test_split(
        self,
        train_prompt_ids: List[str],
        test_prompt_ids: List[str],
    ) -> DataSlice:
        """
        Return a DataSlice with train/test splits by prompt ID.

        Args:
            train_prompt_ids: prompt IDs for the train split
            test_prompt_ids: prompt IDs for the test split
        """
        return DataSlice(
            train_df=self._build_rollout_df(train_prompt_ids),
            test_df=self._build_rollout_df(test_prompt_ids),
        )

    def get_monitor_data(
        self,
        test_prompt_ids: List[str],
        example_prompt_ids: List[str],
        n_examples_per_class: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Prepare data for the LLM monitor.

        For each rollout in test_prompt_ids, create a row containing:
        - The rollout to classify (cot + answer)
        - Few-shot examples grouped by question (examples_by_prompt)
        - Ground truth label

        Args:
            test_prompt_ids: prompts whose rollouts we want to classify
            example_prompt_ids: prompts to draw few-shot examples from
            n_examples_per_class: number of majority/minority examples
                **per train prompt**. With 6 train prompts and n=3,
                total = 6 * 3 * 2 = 36 examples.
        """
        prompts_json = self._load_prompts_json()

        # Build examples grouped by prompt
        examples_by_prompt = []
        for pid in example_prompt_ids:
            rollouts = self._load_rollouts_for_prompt(pid)
            if not rollouts:
                continue
            rollouts = self._compute_labels(rollouts)
            prompt_text = prompts_json.get(pid, "")

            maj = [r for r in rollouts if r["label"] == "majority"]
            mino = [r for r in rollouts if r["label"] == "minority"]
            selected = maj[:n_examples_per_class] + mino[:n_examples_per_class]

            examples = []
            for r in selected:
                examples.append({
                    "cot_content": r.get("cot_content", ""),
                    "answer": r["answer"],
                    "label": r["label"],
                })

            examples_by_prompt.append({
                "prompt_id": pid,
                "prompt_text": prompt_text,
                "examples": examples,
            })

        # Build test rows
        rows = []
        for pid in test_prompt_ids:
            rollouts = self._load_rollouts_for_prompt(pid)
            if not rollouts:
                continue
            rollouts = self._compute_labels(rollouts)
            prompt_text = prompts_json.get(pid, "")

            for r in rollouts:
                rows.append({
                    "prompt_id": pid,
                    "prompt_text": prompt_text,
                    "rollout_idx": r["rollout_idx"],
                    "cot_content": r.get("cot_content", ""),
                    "response_content": r.get("response_content", ""),
                    "answer": r["answer"],
                    "label": r["label"],
                    "is_majority": r["is_majority"],
                    "majority_answer": r["majority_answer"],
                    "majority_frac": r["majority_frac"],
                    "answer_counts": json.dumps(r["answer_counts"]),
                    "examples_by_prompt": examples_by_prompt,
                })

        return rows
