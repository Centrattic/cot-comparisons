"""
CompressedCotTask — CoT compression via middle-50% sentence selection.

Loads existing CoTs from verification rollouts, identifies the middle 50%
of sentences, and prepares data for compression methods (LLM monitor,
thought anchors). Evaluation forces the model with the compressed CoT
and compares answer distributions to the full-CoT baseline.
"""

import contextlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..base import BaseTask
from ...utils.questions import GPQAQuestion, BinaryJudgeQuestion, Question
from src2.tasks.forced_response.prompts import split_cot_into_sentences

# Kimi K2 chat template tokens
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"


@dataclass
class CompressionSpec:
    """Specification for how to compress a CoT's middle section."""
    question_id: str
    rollout_idx: int
    sentences: List[str]
    middle_start: int
    middle_end: int
    target_num_sentences: int
    char_budget: int
    compression_factor: int

    @property
    def middle_sentences(self) -> List[str]:
        return self.sentences[self.middle_start:self.middle_end]

    @property
    def first_quarter(self) -> str:
        return " ".join(self.sentences[:self.middle_start])

    @property
    def last_quarter(self) -> str:
        return " ".join(self.sentences[self.middle_end:])

    @property
    def full_cot(self) -> str:
        return " ".join(self.sentences)

    def reconstruct(self, compressed_middle: str) -> str:
        """Reconstruct full CoT with compressed middle section."""
        parts = []
        if self.first_quarter:
            parts.append(self.first_quarter)
        if compressed_middle:
            parts.append(compressed_middle)
        if self.last_quarter:
            parts.append(self.last_quarter)
        return " ".join(parts)

    def reconstruct_from_indices(self, selected_indices: List[int]) -> str:
        """Reconstruct full CoT keeping only selected middle sentences."""
        middle = self.middle_sentences
        kept = " ".join(middle[i] for i in sorted(selected_indices) if i < len(middle))
        return self.reconstruct(kept)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "rollout_idx": self.rollout_idx,
            "num_sentences": len(self.sentences),
            "middle_start": self.middle_start,
            "middle_end": self.middle_end,
            "num_middle_sentences": len(self.middle_sentences),
            "target_num_sentences": self.target_num_sentences,
            "char_budget": self.char_budget,
            "compression_factor": self.compression_factor,
            "middle_char_length": sum(len(s) for s in self.middle_sentences),
        }


class CompressedCotTask(BaseTask):
    """
    CoT compression task: compress the middle 50% of a CoT ~Nx.

    Loads verified CoTs, splits into sentences, identifies the middle 50%,
    and provides data for compression methods (LLM monitor sentence selection,
    LLM summary rewrite, thought anchors adaptive resampling).

    Evaluation forces the model with the compressed CoT and compares
    answer distributions to the full-CoT baseline.
    """

    def __init__(
        self,
        model: str,
        compression_factor: int = 10,
        char_limit_multiplier: float = 1.5,
        data_dir: Optional[Path] = None,
    ):
        super().__init__("compressed_cot", data_dir)
        self.model = model
        self.compression_factor = compression_factor
        self.char_limit_multiplier = char_limit_multiplier

        self.verification_dir = self.data_dir.parent / "verification_rollouts"
        self.compression_dir = self.data_dir / "compressions"

        for d in [self.verification_dir, self.compression_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(
        self,
        question_id: str,
        rollout_idx: int = 0,
        verbose: bool = True,
    ) -> Optional[CompressionSpec]:
        """
        Load a verified CoT and prepare compression spec.

        This does NOT run model calls — it just prepares the sentence
        breakdown and compression parameters for methods to consume.
        """
        loaded = self.load_question_and_cot(question_id, rollout_idx)
        if loaded is None:
            print(f"Could not load question/CoT for {question_id}")
            return None
        question, source_cot = loaded

        sentences = split_cot_into_sentences(source_cot)
        n = len(sentences)
        if n < 4:
            print(f"CoT too short ({n} sentences) to compress")
            return None

        middle_start = n // 4
        middle_end = (3 * n) // 4
        middle_sentences = sentences[middle_start:middle_end]
        num_middle = len(middle_sentences)

        target_num_sentences = max(1, num_middle // self.compression_factor)
        middle_char_length = sum(len(s) for s in middle_sentences)
        char_budget = int(middle_char_length / self.compression_factor * self.char_limit_multiplier)

        spec = CompressionSpec(
            question_id=question_id,
            rollout_idx=rollout_idx,
            sentences=sentences,
            middle_start=middle_start,
            middle_end=middle_end,
            target_num_sentences=target_num_sentences,
            char_budget=char_budget,
            compression_factor=self.compression_factor,
        )

        # Save spec and question data
        run_dir = self._create_run_dir(question_id, rollout_idx)
        spec_data = spec.to_dict()
        spec_data["sentences"] = sentences
        spec_data["question_type"] = question.question_type
        if isinstance(question, GPQAQuestion):
            spec_data["question"] = question.question
            spec_data["choices"] = question.choices
            spec_data["correct_answer"] = question.correct_answer
        else:
            spec_data["question"] = question.question
            spec_data["bad_outcome"] = question.bad_outcome

        with open(run_dir / "compression_spec.json", "w") as f:
            json.dump(spec_data, f, indent=2)

        if verbose:
            print(f"Compression spec for {question_id}:")
            print(f"  Total sentences: {n}")
            print(f"  Middle 50%: sentences {middle_start}-{middle_end} ({num_middle} sentences)")
            print(f"  Target: {target_num_sentences} sentences, {char_budget} chars")
            print(f"  Saved to {run_dir}")

        return spec

    def get_data(self, load: bool = False) -> Union[bool, Optional[List[Dict]]]:
        if not load:
            return self.compression_dir.exists() and any(
                self.compression_dir.rglob("compression_spec.json")
            )
        specs = sorted(self.compression_dir.rglob("compression_spec.json"))
        if not specs:
            return None
        results = []
        for p in specs:
            with open(p) as f:
                results.append(json.load(f))
        return results

    def get_activations(self, load: bool = False) -> Union[bool, Optional[Any]]:
        # No pre-extracted activations for this task
        if not load:
            return False
        return None

    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Compare predicted vs ground-truth answer distributions via JS divergence."""
        if not predictions or not ground_truth:
            return {"js_divergence": 1.0, "agreement": 0.0}

        js_divs = []
        agreements = []
        for pred, gt in zip(predictions, ground_truth):
            if pred is None or gt is None:
                js_divs.append(1.0)
                agreements.append(0.0)
                continue
            js_divs.append(_js_divergence(pred, gt))
            pred_top = max(pred, key=pred.get) if pred else ""
            gt_top = max(gt, key=gt.get) if gt else ""
            agreements.append(1.0 if pred_top == gt_top else 0.0)

        return {
            "js_divergence": float(np.mean(js_divs)),
            "agreement": float(np.mean(agreements)),
        }

    # ------------------------------------------------------------------
    # Data preparation for methods
    # ------------------------------------------------------------------

    def prepare_for_monitor(self) -> List[Dict]:
        """
        Prepare row dicts for LlmMonitor (sentence selection or summary).

        Returns one row per compression spec, containing all info the
        prompt needs: question, full CoT, numbered middle sentences,
        compression targets, etc.
        """
        data = self.get_data(load=True)
        if data is None:
            raise RuntimeError("No data found. Run run_data() first.")

        rows = []
        for spec_data in data:
            sentences = spec_data["sentences"]
            middle_start = spec_data["middle_start"]
            middle_end = spec_data["middle_end"]
            middle_sentences = sentences[middle_start:middle_end]

            row = {
                "question_id": spec_data["question_id"],
                "question_type": spec_data.get("question_type", "multiple_choice"),
                "question": spec_data.get("question", ""),
                "full_cot": " ".join(sentences),
                "sentences": sentences,
                "middle_sentences": middle_sentences,
                "middle_start_idx": middle_start,
                "middle_end_idx": middle_end,
                "target_num_sentences": spec_data["target_num_sentences"],
                "char_budget": spec_data["char_budget"],
                "compression_factor": spec_data["compression_factor"],
                "rollout_idx": spec_data.get("rollout_idx", 0),
            }
            if "choices" in spec_data:
                row["choices"] = spec_data["choices"]
            if "correct_answer" in spec_data:
                row["correct_answer"] = spec_data["correct_answer"]
            if "bad_outcome" in spec_data:
                row["bad_outcome"] = spec_data["bad_outcome"]
            rows.append(row)
        return rows

    def prepare_for_thought_anchors(self) -> List[Dict]:
        """
        Prepare data for ThoughtAnchors method.

        Returns one row per compression spec with the same data as
        prepare_for_monitor, plus the full sentence list for
        sentence-removal resampling.
        """
        return self.prepare_for_monitor()

    # ------------------------------------------------------------------
    # Evaluation: force model with compressed CoT
    # ------------------------------------------------------------------

    def evaluate_compression(
        self,
        question_id: str,
        rollout_idx: int,
        compressed_cot: str,
        num_resamples: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Force model with compressed CoT and return answer distribution.

        Runs num_resamples continuations from the compressed CoT prefix
        via Tinker, extracts answers, and returns the distribution.
        """
        from tinker import ServiceClient, types
        from transformers import AutoTokenizer

        loaded = self.load_question_and_cot(question_id, rollout_idx)
        if loaded is None:
            raise RuntimeError(f"Could not load question for {question_id}")
        question, _ = loaded

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        client = ServiceClient()
        sampling_client = client.create_sampling_client(base_model=self.model)
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)

        prompt_str = (
            f"{IM_SYSTEM}system{IM_MIDDLE}You are Kimi, an AI assistant created by Moonshot AI.{IM_END}"
            f"{IM_USER}user{IM_MIDDLE}{self._user_msg(question)}{IM_END}"
            f"{IM_ASSISTANT}assistant{IM_MIDDLE}<think>{compressed_cot}"
        )

        def run_single(idx: int) -> Optional[str]:
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            model_input = types.ModelInput.from_ints(tokens)
            result = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=params,
            ).result()
            sample_tokens = result.sequences[0].tokens
            answer, _, _ = self._extract_answer(sample_tokens, tokenizer, question)
            return answer if answer else None

        answers = []
        with ThreadPoolExecutor(max_workers=min(300, num_resamples)) as executor:
            futures = {executor.submit(run_single, i): i for i in range(num_resamples)}
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Evaluating compression", disable=not verbose,
            ):
                ans = future.result()
                if ans:
                    answers.append(ans)

        counts = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        total = len(answers)
        distribution = {k: v / total for k, v in counts.items()} if total > 0 else {}

        most_common = max(counts.items(), key=lambda x: x[1]) if counts else ("", 0)

        return {
            "question_id": question_id,
            "num_resamples": num_resamples,
            "valid_answers": total,
            "answer_counts": counts,
            "distribution": distribution,
            "most_common": most_common[0],
            "agreement_rate": most_common[1] / total if total > 0 else 0,
        }

    def get_baseline_distribution(
        self,
        question_id: str,
        rollout_idx: int = 0,
        num_resamples: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Get baseline answer distribution by forcing with the full CoT.
        """
        loaded = self.load_question_and_cot(question_id, rollout_idx)
        if loaded is None:
            raise RuntimeError(f"Could not load question for {question_id}")
        _, source_cot = loaded
        return self.evaluate_compression(
            question_id, rollout_idx, source_cot,
            num_resamples=num_resamples, temperature=temperature,
            max_tokens=max_tokens, verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Question/CoT loading helpers
    # ------------------------------------------------------------------

    def load_question_and_cot(
        self, question_id: str, rollout_idx: int = 0,
    ) -> Optional[Tuple[Question, str]]:
        """Load a Question object and its source CoT from verification data."""
        verification_dir = self.verification_dir / question_id
        if not verification_dir.exists():
            return None

        run_dir = self._get_latest_verification_dir(question_id)
        if run_dir is None:
            return None

        rollout_path = run_dir / "rollouts" / f"rollout_{rollout_idx:03d}.json"
        summary_path = run_dir / "summary.json"

        if not rollout_path.exists() or not summary_path.exists():
            return None

        with open(rollout_path) as f:
            rollout_data = json.load(f)
        with open(summary_path) as f:
            summary = json.load(f)

        question = self._question_from_summary(summary)
        source_cot = rollout_data.get("thinking", "") or rollout_data.get("full_response", "")
        if not source_cot:
            return None

        return question, source_cot

    def get_verified_questions(self, threshold: float = 0.8) -> List[str]:
        """Get question IDs that meet the verification agreement threshold."""
        verified = []
        if not self.verification_dir.exists():
            return verified
        for qdir in self.verification_dir.iterdir():
            if qdir.is_dir():
                summary = self._load_verification_summary(qdir.name)
                if summary and summary.get("agreement_rate", 0) >= threshold:
                    verified.append(qdir.name)
        return verified

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _create_run_dir(self, question_id: str, rollout_idx: int) -> Path:
        base = self.compression_dir / question_id / f"rollout_{rollout_idx:03d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_latest_verification_dir(self, question_id: str) -> Optional[Path]:
        qdir = self.verification_dir / question_id
        if not qdir.exists():
            return None
        timestamped = sorted(
            [d for d in qdir.iterdir()
             if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
            reverse=True,
        )
        if timestamped:
            return timestamped[0]
        if (qdir / "summary.json").exists():
            return qdir
        return None

    def _load_verification_summary(self, question_id: str) -> Optional[Dict]:
        run_dir = self._get_latest_verification_dir(question_id)
        if run_dir:
            path = run_dir / "summary.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        return None

    # ------------------------------------------------------------------
    # Internal helpers (shared with ResamplingTask)
    # ------------------------------------------------------------------

    @staticmethod
    def _question_from_summary(summary: Dict) -> Question:
        qt = summary.get("question_type", "multiple_choice")
        if qt == "binary_judge":
            return BinaryJudgeQuestion(
                id=summary["question_id"], question=summary["question"],
                judge_prompt=summary["judge_prompt"], bad_outcome=summary["bad_outcome"],
                subject=summary.get("subject"),
            )
        return GPQAQuestion(
            id=summary["question_id"], question=summary["question"],
            choices=summary["choices"], correct_answer=summary["correct_answer"],
            correct_index=ord(summary["correct_answer"]) - ord("A"),
        )

    @staticmethod
    def _user_msg(question: Question) -> str:
        if isinstance(question, BinaryJudgeQuestion):
            return question.question
        labels = [chr(ord("A") + i) for i in range(len(question.choices))]
        choices = "\n".join(f"{l}. {c}" for l, c in zip(labels, question.choices))
        return f"{question.question}\n\n{choices}\n\nAnswer with just the letter (A, B, C, or D)."

    @staticmethod
    def _extract_answer(tokens, tokenizer, question: Question):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if "</think>" in text:
            parts = text.split("</think>", 1)
            continued_cot = parts[0]
            answer_text = parts[1].strip()
        else:
            continued_cot = text
            answer_text = ""

        if isinstance(question, BinaryJudgeQuestion):
            upper = answer_text.upper()
            if "YES" in upper:
                return "YES", continued_cot, text
            if "NO" in upper:
                return "NO", continued_cot, text
            return "", continued_cot, text

        clean = answer_text.upper().rstrip(".").strip()
        if clean in ["A", "B", "C", "D"]:
            return clean, continued_cot, text
        match = re.search(r"\b([A-D])\b", answer_text)
        if match:
            return match.group(1).upper(), continued_cot, text
        return "", continued_cot, text


# ------------------------------------------------------------------
# Utility: JS divergence for comparing distributions
# ------------------------------------------------------------------

def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two distributions (as dicts)."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 1.0

    p_arr = np.array([p.get(k, 0.0) for k in sorted(all_keys)])
    q_arr = np.array([q.get(k, 0.0) for k in sorted(all_keys)])

    # Normalize
    p_sum, q_sum = p_arr.sum(), q_arr.sum()
    if p_sum > 0:
        p_arr = p_arr / p_sum
    if q_sum > 0:
        q_arr = q_arr / q_sum

    m = 0.5 * (p_arr + q_arr)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p_arr, m) + 0.5 * _kl(q_arr, m)
