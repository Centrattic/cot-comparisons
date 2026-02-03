"""
CompressedCotTask — CoT compression via configurable region selection.

Loads existing CoTs from verification rollouts, identifies a configurable
region (prefix or middle) of sentences, and prepares data for compression
methods (LLM monitor, thought anchors). Evaluation forces the model with
the compressed CoT and compares answer distributions to the full-CoT baseline.

Supports two region types:
  - "prefix": compress the first compress_pct of sentences
  - "middle": compress the middle compress_pct of sentences
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
    """Specification for how to compress a CoT region."""
    question_id: str
    rollout_idx: int
    sentences: List[str]
    region_start: int
    region_end: int
    region_type: str  # "prefix" or "middle"
    target_num_sentences: int
    char_budget: int
    compression_factor: int
    original_token_count: int  # total token count of the original CoT
    region_token_count: int  # token count of the original uncompressed region

    @property
    def region_sentences(self) -> List[str]:
        return self.sentences[self.region_start:self.region_end]

    @property
    def pre_region(self) -> str:
        return " ".join(self.sentences[:self.region_start])

    @property
    def post_region(self) -> str:
        return " ".join(self.sentences[self.region_end:])

    @property
    def full_cot(self) -> str:
        return " ".join(self.sentences)

    def reconstruct(self, compressed_region: str) -> str:
        """Reconstruct full CoT with compressed region."""
        parts = []
        if self.pre_region:
            parts.append(self.pre_region)
        if compressed_region:
            parts.append(compressed_region)
        if self.post_region:
            parts.append(self.post_region)
        return " ".join(parts)

    def reconstruct_from_indices(self, selected_indices: List[int]) -> str:
        """Reconstruct full CoT keeping only selected region sentences."""
        region = self.region_sentences
        kept = " ".join(region[i] for i in sorted(selected_indices) if i < len(region))
        return self.reconstruct(kept)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "rollout_idx": self.rollout_idx,
            "num_sentences": len(self.sentences),
            "region_start": self.region_start,
            "region_end": self.region_end,
            "region_type": self.region_type,
            "num_region_sentences": len(self.region_sentences),
            "target_num_sentences": self.target_num_sentences,
            "char_budget": self.char_budget,
            "compression_factor": self.compression_factor,
            "original_token_count": self.original_token_count,
            "region_token_count": self.region_token_count,
            "region_char_length": sum(len(s) for s in self.region_sentences),
        }


class CompressedCotTask(BaseTask):
    """
    CoT compression task: compress a configurable region of a CoT ~Nx.

    Loads verified CoTs, splits into sentences, identifies the region to
    compress (prefix or middle), and provides data for compression methods
    (LLM monitor sentence selection, LLM summary rewrite, thought anchors
    adaptive resampling).

    Evaluation forces the model with the compressed CoT and compares
    answer distributions to the full-CoT baseline.
    """

    def __init__(
        self,
        model: str,
        compression_factor: int = 10,
        char_limit_multiplier: float = 1.5,
        compress_pct: float = 0.5,
        region: str = "prefix",
        data_dir: Optional[Path] = None,
    ):
        super().__init__("compressed_cot", data_dir)
        self.model = model
        self.compression_factor = compression_factor
        self.char_limit_multiplier = char_limit_multiplier
        self.compress_pct = compress_pct
        self.region = region

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

        # Compute region boundaries based on region type
        if self.region == "prefix":
            region_start = 0
            region_end = int(n * self.compress_pct)
        elif self.region == "middle":
            region_start = int(n * (1 - self.compress_pct) / 2)
            region_end = int(n * (1 + self.compress_pct) / 2)
        else:
            raise ValueError(f"Unknown region type: {self.region!r}. Use 'prefix' or 'middle'.")

        # Ensure at least 1 sentence in region
        if region_end <= region_start:
            region_end = region_start + 1

        region_sentences = sentences[region_start:region_end]
        num_region = len(region_sentences)

        target_num_sentences = max(1, num_region // self.compression_factor)
        region_char_length = sum(len(s) for s in region_sentences)
        char_budget = int(region_char_length / self.compression_factor * self.char_limit_multiplier)

        # Tokenize for token budget computation
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        with contextlib.redirect_stdout(io.StringIO()):
            original_tokens = tokenizer.encode(source_cot, add_special_tokens=False)
            region_text = " ".join(region_sentences)
            region_tokens = tokenizer.encode(region_text, add_special_tokens=False)

        original_token_count = len(original_tokens)
        region_token_count = len(region_tokens)

        spec = CompressionSpec(
            question_id=question_id,
            rollout_idx=rollout_idx,
            sentences=sentences,
            region_start=region_start,
            region_end=region_end,
            region_type=self.region,
            target_num_sentences=target_num_sentences,
            char_budget=char_budget,
            compression_factor=self.compression_factor,
            original_token_count=original_token_count,
            region_token_count=region_token_count,
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
            region_label = "Prefix" if self.region == "prefix" else "Middle"
            print(f"Compression spec for {question_id}:")
            print(f"  Total sentences: {n}")
            print(f"  {region_label} {self.compress_pct:.0%}: sentences {region_start}-{region_end} ({num_region} sentences)")
            print(f"  Original tokens: {original_token_count}, Region tokens: {region_token_count}")
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
        prompt needs: question, full CoT, numbered region sentences,
        compression targets, etc.
        """
        data = self.get_data(load=True)
        if data is None:
            raise RuntimeError("No data found. Run run_data() first.")

        rows = []
        for spec_data in data:
            sentences = spec_data["sentences"]
            region_start = spec_data.get("region_start", spec_data.get("middle_start", 0))
            region_end = spec_data.get("region_end", spec_data.get("middle_end", len(sentences)))
            region_type = spec_data.get("region_type", "middle")
            region_sentences = sentences[region_start:region_end]

            row = {
                "question_id": spec_data["question_id"],
                "question_type": spec_data.get("question_type", "multiple_choice"),
                "question": spec_data.get("question", ""),
                "full_cot": " ".join(sentences),
                "sentences": sentences,
                "region_sentences": region_sentences,
                "region_start_idx": region_start,
                "region_end_idx": region_end,
                "region_type": region_type,
                "target_num_sentences": spec_data["target_num_sentences"],
                "char_budget": spec_data["char_budget"],
                "compression_factor": spec_data["compression_factor"],
                "rollout_idx": spec_data.get("rollout_idx", 0),
                "original_token_count": spec_data.get("original_token_count", 0),
                "region_token_count": spec_data.get("region_token_count", 0),
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
        num_resamples: int = 50,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Force model with compressed CoT and return answer distribution.

        For prefix region_type: two-step generation —
          1. Force compressed prefix, generate exactly continuation_budget
             thinking tokens (original_token_count - region_token_count).
          2. Append </think> to the output, then generate the answer.

        For middle region_type: forces with pre_region + compressed_middle +
        post_region, uses the provided max_tokens (original behavior).

        Runs num_resamples continuations, extracts answers, returns distribution.
        """
        from tinker import ServiceClient, types
        from transformers import AutoTokenizer

        loaded = self.load_question_and_cot(question_id, rollout_idx)
        if loaded is None:
            raise RuntimeError(f"Could not load question for {question_id}")
        question, _ = loaded

        # Load the spec to get region_type and token counts
        spec_data = self._load_latest_spec(question_id, rollout_idx)
        region_type = spec_data.get("region_type", "middle") if spec_data else "middle"

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        client = ServiceClient()
        sampling_client = client.create_sampling_client(base_model=self.model)

        # Compute continuation budget for prefix mode
        continuation_budget = None
        if region_type == "prefix" and spec_data:
            original_token_count = spec_data.get("original_token_count", 0)
            region_token_count = spec_data.get("region_token_count", 0)
            if original_token_count > 0 and region_token_count > 0:
                continuation_budget = original_token_count - region_token_count

        prompt_str = (
            f"{IM_SYSTEM}system{IM_MIDDLE}You are Kimi, an AI assistant created by Moonshot AI.{IM_END}"
            f"{IM_USER}user{IM_MIDDLE}{self._user_msg(question)}{IM_END}"
            f"{IM_ASSISTANT}assistant{IM_MIDDLE}<think>{compressed_cot}"
        )

        # Precompute </think> tokens for prefix two-step generation
        with contextlib.redirect_stdout(io.StringIO()):
            end_think_tokens = tokenizer.encode("</think>", add_special_tokens=False)

        ANSWER_MAX_TOKENS = 256

        def run_single(idx: int) -> Optional[str]:
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

            if region_type == "prefix" and continuation_budget is not None:
                # Step 1: generate exactly continuation_budget thinking tokens
                think_params = types.SamplingParams(
                    max_tokens=continuation_budget, temperature=temperature,
                )
                model_input = types.ModelInput.from_ints(tokens)
                think_result = sampling_client.sample(
                    prompt=model_input, num_samples=1, sampling_params=think_params,
                ).result()
                think_tokens = list(think_result.sequences[0].tokens)

                # Step 2: append </think> and generate the answer
                full_prefix_tokens = tokens + think_tokens + end_think_tokens
                answer_params = types.SamplingParams(
                    max_tokens=ANSWER_MAX_TOKENS, temperature=temperature,
                )
                answer_input = types.ModelInput.from_ints(full_prefix_tokens)
                answer_result = sampling_client.sample(
                    prompt=answer_input, num_samples=1, sampling_params=answer_params,
                ).result()
                answer_tokens = answer_result.sequences[0].tokens

                # Extract answer from the response-only tokens (after </think>)
                answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                answer = self._extract_answer_from_text(answer_text, question)
                return answer if answer else None
            else:
                # Middle mode: single-step generation (original behavior)
                params = types.SamplingParams(
                    max_tokens=max_tokens, temperature=temperature,
                )
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
            "region_type": region_type,
            "continuation_budget": continuation_budget,
        }

    def get_baseline_distribution(
        self,
        question_id: str,
        rollout_idx: int = 0,
        num_resamples: int = 50,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Get baseline answer distribution by forcing with the full CoT.

        For prefix mode, forces with the full original CoT at the same
        token budget to get baseline distribution.
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
    # End-to-end evaluation from monitor results
    # ------------------------------------------------------------------

    def evaluate_monitor_results(
        self,
        monitor_results: List[Dict],
        spec: CompressionSpec,
        output_folder: Path,
        num_resamples: int = 50,
        temperature: float = 0.7,
        verbose: bool = True,
        baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate compression from monitor sentence-selection results.

        Takes the first monitor result, converts selected indices to
        region-relative, reconstructs the compressed CoT, runs evaluation
        resamples, and saves results to output_folder.

        If baseline is provided, it's used directly instead of recomputing.
        """
        # Extract selected indices from monitor prediction
        selected_indices = monitor_results[0].get("monitor_prediction", [])
        if not isinstance(selected_indices, list):
            selected_indices = []

        # Convert absolute indices to region-relative
        relative_indices = [i - spec.region_start for i in selected_indices]

        # Reconstruct compressed CoT
        compressed_cot = spec.reconstruct_from_indices(relative_indices)

        # Evaluate compressed distribution
        compressed_dist = self.evaluate_compression(
            spec.question_id, spec.rollout_idx, compressed_cot,
            num_resamples=num_resamples, temperature=temperature, verbose=verbose,
        )

        # Use provided baseline or compute it
        baseline_dist = baseline if baseline is not None else self.get_baseline_distribution(
            spec.question_id, spec.rollout_idx,
            num_resamples=num_resamples, temperature=temperature, verbose=verbose,
        )

        # Compute metrics
        metrics = self.evaluate(
            [compressed_dist["distribution"]], [baseline_dist["distribution"]],
        )

        eval_results = {
            "selected_indices": selected_indices,
            "relative_indices": relative_indices,
            "compressed_distribution": compressed_dist,
            "baseline_distribution": baseline_dist,
            **metrics,
        }

        # Save outputs
        output_folder = Path(output_folder)
        with open(output_folder / "compression_eval.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        with open(output_folder / "compressed_cot.txt", "w") as f:
            f.write(compressed_cot)

        if verbose:
            print(f"JS divergence: {metrics['js_divergence']:.4f}")
            print(f"Agreement: {metrics['agreement']:.1%}")
            print(f"Saved to {output_folder}")

        return eval_results

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

    def _load_latest_spec(self, question_id: str, rollout_idx: int) -> Optional[Dict]:
        """Load the latest compression_spec.json for a question/rollout."""
        base = self.compression_dir / question_id / f"rollout_{rollout_idx:03d}"
        if not base.exists():
            return None
        specs = sorted(base.rglob("compression_spec.json"), reverse=True)
        if not specs:
            return None
        with open(specs[0]) as f:
            return json.load(f)

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
    def _extract_answer_from_text(answer_text: str, question: Question) -> str:
        """Extract answer from already-decoded text (after </think>)."""
        if isinstance(question, BinaryJudgeQuestion):
            upper = answer_text.upper()
            if "YES" in upper:
                return "YES"
            if "NO" in upper:
                return "NO"
            return ""
        clean = answer_text.upper().rstrip(".").strip()
        if clean in ["A", "B", "C", "D"]:
            return clean
        match = re.search(r"\b([A-D])\b", answer_text)
        if match:
            return match.group(1).upper()
        return ""

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
