"""
ResamplingTask — evenly-spaced prefix continuation via Tinker.

Standalone task (no shared base class beyond BaseTask). Selects ~N
evenly-spaced prefix points from a source CoT and runs M independent
continuations at each point to get answer distributions.
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
from ...data_slice import DataSlice
from ...utils.questions import GPQAQuestion, BinaryJudgeQuestion, Question
from src2.tasks.forced_response.prompts import get_cumulative_cot_segments

# Kimi K2 chat template tokens
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"


@dataclass
class ResampleResult:
    """Result of a single resample attempt."""
    sentence_idx: int
    resample_idx: int
    forced_prefix: str
    continuation: str
    full_response: str
    answer: str
    raw_tokens: List[int]
    full_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx, "resample_idx": self.resample_idx,
            "forced_prefix": self.forced_prefix, "continuation": self.continuation,
            "full_response": self.full_response, "answer": self.answer,
            "raw_tokens": self.raw_tokens, "full_prompt": self.full_prompt,
        }


class ResamplingTask(BaseTask):
    """
    Resampling task: evenly-spaced prefix continuations via Tinker.

    run_data() selects ~num_prefix_points evenly-spaced prefixes, then
    runs num_resamples continuations at each to produce answer distributions.
    """

    def __init__(self, model: str,
                 data_dir: Optional[Path] = None):
        super().__init__("resampling", data_dir or (
            Path(__file__).parent.parent.parent / "data" / "forced_response"
        ))
        self.model = model

        # Sub-directories
        self.verification_dir = self.data_dir.parent / "verification_rollouts"
        self.resampling_dir = self.data_dir / "resampling"
        self.monitor_resampling_dir = self.data_dir / "monitor_resampling"

        for d in [self.verification_dir, self.resampling_dir, self.monitor_resampling_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(
        self,
        question_id: str,
        rollout_idx: int = 0,
        num_resamples: int = 20,
        num_prefix_points: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Run resampling for evenly-spaced prefix points via Tinker."""
        from tinker import ServiceClient, types
        from transformers import AutoTokenizer

        loaded = self.load_question_and_cot(question_id, rollout_idx)
        if loaded is None:
            print(f"Could not load question/CoT for {question_id}")
            return None
        question, source_cot = loaded

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        client = ServiceClient()
        sampling_client = client.create_sampling_client(base_model=self.model)

        cot_segments = get_cumulative_cot_segments(source_cot)
        num_sentences = len(cot_segments)

        stride = max(num_sentences // num_prefix_points, 1)
        selected_indices = list(range(0, num_sentences, stride))

        config = {
            "model": self.model, "question_id": question.id,
            "rollout_idx": rollout_idx, "num_resamples": num_resamples,
            "num_prefix_points": len(selected_indices), "stride": stride,
            "temperature": temperature, "max_tokens": max_tokens,
            "num_sentences": num_sentences,
        }
        run_dir = self.create_run_dir("resampling", question.id, rollout_idx, config)

        if verbose:
            print(f"Resampling {question.id}: {len(selected_indices)} prefixes × {num_resamples} resamples")
            print(f"Stride: {stride}, run dir: {run_dir}")

        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)

        def resample_single(sent_idx: int, resample_idx: int, partial_cot: str) -> Optional[ResampleResult]:
            prompt_str = (
                f"{IM_SYSTEM}system{IM_MIDDLE}You are Kimi, an AI assistant created by Moonshot AI.{IM_END}"
                f"{IM_USER}user{IM_MIDDLE}{self._user_msg(question)}{IM_END}"
                f"{IM_ASSISTANT}assistant{IM_MIDDLE}<think>{partial_cot}"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            model_input = types.ModelInput.from_ints(tokens)

            result = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=params).result()
            sample_tokens = result.sequences[0].tokens
            answer, continued_cot, full_text = self._extract_answer(sample_tokens, tokenizer, question)

            if answer:
                return ResampleResult(
                    sentence_idx=sent_idx, resample_idx=resample_idx,
                    forced_prefix=partial_cot, continuation=continued_cot,
                    full_response=partial_cot + full_text, answer=answer,
                    raw_tokens=list(sample_tokens), full_prompt=prompt_str,
                )
            return None

        all_tasks = [
            (si, ri, cot_segments[si])
            for si in selected_indices for ri in range(num_resamples)
        ]
        all_results: List[ResampleResult] = []

        with ThreadPoolExecutor(max_workers=min(300, len(all_tasks))) as executor:
            futures = {executor.submit(resample_single, si, ri, pc): (si, ri) for si, ri, pc in all_tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Resampling", disable=not verbose):
                r = future.result()
                if r:
                    all_results.append(r)

        by_sentence: Dict[int, List[ResampleResult]] = {}
        for r in all_results:
            by_sentence.setdefault(r.sentence_idx, []).append(r)

        all_summaries = []
        for si in selected_indices:
            sent_results = by_sentence.get(si, [])
            self._save_resampling_result(
                question=question, sentence_idx=si,
                forced_prefix=cot_segments[si],
                resample_results=[r.to_dict() for r in sent_results],
                run_dir=run_dir,
            )
            counts = {}
            for r in sent_results:
                counts[r.answer] = counts.get(r.answer, 0) + 1
            total_valid = len(sent_results)
            most_common = max(counts.items(), key=lambda x: x[1]) if counts else ("", 0)
            all_summaries.append({
                "sentence_idx": si, "forced_prefix_length": len(cot_segments[si]),
                "total_resamples": total_valid, "valid_answers": total_valid,
                "answer_counts": counts, "most_common": most_common[0],
                "most_common_count": most_common[1],
                "agreement_rate": most_common[1] / total_valid if total_valid > 0 else 0,
            })

        self._save_resampling_summary(question, rollout_idx, all_summaries, run_dir)

        if verbose:
            print(f"Done: {len(selected_indices)} prefix points processed.")

        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "num_sentences": num_sentences, "num_prefix_points": len(selected_indices),
            "stride": stride, "num_resamples": num_resamples,
            "sentence_results": all_summaries,
        }
        if isinstance(question, GPQAQuestion):
            summary["correct_answer"] = question.correct_answer
        else:
            summary["bad_outcome"] = question.bad_outcome
        return summary

    def get_data(self, load: bool = False) -> Union[bool, Optional[Any]]:
        if not load:
            return self.resampling_dir.exists() and any(self.resampling_dir.rglob("summary.json"))
        summaries = sorted(self.resampling_dir.rglob("summary.json"))
        if not summaries:
            return None
        results = []
        for p in summaries:
            with open(p) as f:
                results.append(json.load(f))
        return results

    def get_activations(self, load: bool = False) -> Union[bool, Optional[List[Path]]]:
        if not load:
            return self.resampling_dir.exists() and any(self.resampling_dir.rglob("resample_*.json"))
        paths = sorted(self.resampling_dir.rglob("resample_*.json"))
        return paths if paths else None

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        if not predictions or not ground_truth:
            return {"accuracy": 0.0}
        correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p).upper() == str(g).upper())
        return {"accuracy": correct / len(predictions)}

    # ------------------------------------------------------------------
    # Data preparation for methods
    # ------------------------------------------------------------------

    def prepare_for_probe(
        self, extractor, layer: int,
        data_slice: DataSlice,
        token_position: str = "last_thinking",
    ) -> Dict[str, Any]:
        """Prepare activation data for LinearProbe (soft_ce mode)."""
        data = self.get_data(load=True)
        if data is None:
            raise RuntimeError("No data found. Run run_data() first.")

        samples = []
        for summary in data:
            question_id = summary["question_id"]
            if not data_slice.matches_id(question_id):
                continue
            for sent in summary.get("sentence_summaries", []):
                sentence_idx = sent["sentence_idx"]
                if not data_slice.matches_sentence(sentence_idx):
                    continue
                counts = sent.get("answer_counts", {})
                total = sum(counts.values())
                if total == 0:
                    continue
                dist = {k: v / total for k, v in counts.items()}
                samples.append({
                    "question_id": question_id,
                    "sentence_idx": sentence_idx,
                    "answer_distribution": dist,
                    "question_type": summary.get("question_type", "multiple_choice"),
                })
        return samples

    def prepare_for_monitor(self, data_slice: DataSlice) -> List[Dict]:
        """Prepare row dicts for LlmMonitor."""
        data = self.get_data(load=True)
        if data is None:
            raise RuntimeError("No data found. Run run_data() first.")

        rows = []
        for summary in data:
            question_id = summary["question_id"]
            if not data_slice.matches_id(question_id):
                continue
            for sent in summary.get("sentence_summaries", []):
                sentence_idx = sent["sentence_idx"]
                if not data_slice.matches_sentence(sentence_idx):
                    continue
                row = {
                    "question_id": question_id,
                    "question_type": summary.get("question_type", "multiple_choice"),
                    "partial_cot": sent.get("forced_prefix", ""),
                    "sentence_idx": sentence_idx,
                }
                if "correct_answer" in summary:
                    row["correct_answer"] = summary["correct_answer"]
                if "bad_outcome" in summary:
                    row["bad_outcome"] = summary["bad_outcome"]
                rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Activation extraction and data serving
    # ------------------------------------------------------------------

    def extract_activations(
        self,
        model_name: str,
        layer: int,
        data_slice: DataSlice,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Extract and save full-sequence activations for resampling runs.

        Walks all resample_*.json files, extracts full [seq_len, hidden_dim]
        activations plus token boundary indices, and saves them as
        companion .npz files.
        """
        from ...utils.activations import ActivationExtractor, compute_token_boundaries

        if not self.resampling_dir.exists():
            raise RuntimeError(f"No resampling data found at {self.resampling_dir}")

        run_files = sorted(self.resampling_dir.rglob("resample_*.json"))
        if not run_files:
            raise RuntimeError(f"No resample run files found in {self.resampling_dir}")

        run_files = data_slice.filter_paths(run_files)
        filtered_files = []
        for rp in run_files:
            parts = rp.parts
            question_id = None
            sentence_idx = None
            for part in parts:
                if part.startswith("sentence_"):
                    try:
                        sentence_idx = int(part.split("_", 1)[1])
                    except ValueError:
                        pass
            try:
                resamp_idx = parts.index("resampling")
                if resamp_idx + 1 < len(parts):
                    question_id = parts[resamp_idx + 1]
            except ValueError:
                pass
            if question_id is not None and not data_slice.matches_id(question_id):
                continue
            if sentence_idx is not None and not data_slice.matches_sentence(sentence_idx):
                continue
            filtered_files.append(rp)
        run_files = filtered_files

        extractor = ActivationExtractor(
            model_name=model_name, load_in_4bit=load_in_4bit,
        )
        seq_key = f"layer{layer}_full_sequence"
        bnd_key = f"layer{layer}_boundaries"

        for run_path in tqdm(run_files, desc="Extracting activations (resampling)"):
            act_path = run_path.with_suffix(".npz")
            if act_path.exists():
                with np.load(act_path) as f:
                    if seq_key in f.files:
                        continue

            with open(run_path) as f:
                run_data = json.load(f)

            full_response = run_data.get("full_response", "")
            if not full_response:
                continue

            full_prompt = run_data.get("full_prompt", "")
            if not full_prompt:
                import logging
                logging.warning(
                    "No full_prompt in %s — skipping (old data without full context)",
                    run_path,
                )
                continue

            full_text = full_prompt + full_response

            try:
                act = extractor.extract_full_sequence(full_text, layer)
                boundaries = compute_token_boundaries(
                    extractor.tokenizer, full_prompt, full_response,
                )
                bnd_array = np.array([
                    boundaries["last_input"],
                    boundaries["last_thinking"],
                    boundaries["last_response"],
                ], dtype=np.int64)

                arrays = {}
                if act_path.exists():
                    with np.load(act_path) as f:
                        for k in f.files:
                            arrays[k] = f[k]
                arrays[seq_key] = act
                arrays[bnd_key] = bnd_array
                np.savez(act_path, **arrays)
            except Exception:
                continue

        print(f"Activations extracted for resampling data in {self.resampling_dir}")

    def get_probe_data(
        self,
        layer: int,
        data_slice: DataSlice,
        token_position: str = "last_thinking",
    ) -> List[Dict]:
        """
        Load pre-extracted activations formatted for probes.

        Args:
            layer: Model layer to load activations from.
            data_slice: Filter for question IDs and sentence indices.
            token_position: One of "last_input", "last_thinking",
                "last_response" (returns [hidden_dim] sliced from full
                sequence), or "full_sequence" (returns [seq_len, hidden_dim]).

        Returns list of dicts:
            {
                "activation": np.ndarray [hidden_dim] or [seq_len, hidden_dim],
                "answer_distribution": {"A": 0.3, "B": 0.5, ...},
                "question_id": str,
                "sentence_idx": int,
                "forced_prefix": str,
                "continuation": str,
                "full_response": str,
                "full_prompt": str,
                "answer": str,
            }
        """
        seq_key = f"layer{layer}_full_sequence"
        bnd_key = f"layer{layer}_boundaries"
        legacy_key = f"layer{layer}_{token_position}"
        boundary_names = ["last_input", "last_thinking", "last_response"]

        # Load sentence-level summaries to get answer distributions
        summary_files = sorted(self.resampling_dir.rglob("sentence_*/summary.json"))
        summary_files = data_slice.filter_paths(summary_files)

        samples = []
        for summary_path in summary_files:
            sentence_dir = summary_path.parent
            with open(summary_path) as f:
                summary = json.load(f)

            question_id = summary.get("question_id", "")
            sentence_idx = summary.get("sentence_idx", -1)
            if not data_slice.matches_id(question_id):
                continue
            if not data_slice.matches_sentence(sentence_idx):
                continue
            answer_counts = summary.get("answer_counts", {})
            total = sum(answer_counts.values())
            if total == 0:
                continue

            answer_distribution = {k: v / total for k, v in answer_counts.items()}

            # Find companion activation files
            act_files = sorted(sentence_dir.glob("resample_*.npz"))
            for act_path in act_files:
                with np.load(act_path) as f:
                    if seq_key in f.files:
                        full_seq = f[seq_key]
                        boundaries = f[bnd_key]
                        if token_position == "full_sequence":
                            activation = full_seq
                        elif token_position in boundary_names:
                            idx = boundaries[boundary_names.index(token_position)]
                            if idx < 0:
                                continue
                            activation = full_seq[idx]
                        else:
                            continue
                    elif legacy_key in f.files:
                        # Backwards compat: old single-token extractions
                        activation = f[legacy_key]
                    else:
                        continue

                # Load companion JSON for text data
                json_path = act_path.with_suffix(".json")
                text_data = {}
                if json_path.exists():
                    with open(json_path) as f:
                        text_data = json.load(f)

                samples.append({
                    "activation": activation,
                    "answer_distribution": answer_distribution,
                    "question_id": question_id,
                    "sentence_idx": sentence_idx,
                    "forced_prefix": text_data.get("forced_prefix", ""),
                    "continuation": text_data.get("continuation", ""),
                    "full_response": text_data.get("full_response", ""),
                    "full_prompt": text_data.get("full_prompt", ""),
                    "answer": text_data.get("answer", ""),
                })

        return samples

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

    def get_question_dir(self, question_id: str, mode: str) -> Path:
        mode_dirs = {
            "verification": self.verification_dir,
            "resampling": self.resampling_dir,
            "monitor_resampling": self.monitor_resampling_dir,
        }
        if mode not in mode_dirs:
            raise ValueError(f"Unknown mode: {mode}")
        d = mode_dirs[mode] / question_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_run_dir(self, mode: str, question_id: str,
                       rollout_idx: int, config: dict) -> Path:
        base = self.get_question_dir(question_id, mode)
        rollout_dir = base / f"rollout_{rollout_idx:03d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = rollout_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        config["timestamp"] = datetime.now().isoformat()
        config["run_type"] = mode
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return run_dir

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------

    def _save_resampling_result(self, question: Question, sentence_idx: int,
                                forced_prefix: str, resample_results: List[Dict],
                                run_dir: Path) -> Path:
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(resample_results):
            with open(sentence_dir / f"resample_{i:03d}.json", "w") as f:
                json.dump(result, f, indent=2)

        answers = [r.get("answer", "").upper() for r in resample_results]
        if isinstance(question, BinaryJudgeQuestion):
            valid = [a for a in answers if a in ["YES", "NO"]]
        else:
            valid = [a for a in answers if a in ["A", "B", "C", "D"]]
        answer_counts = {}
        for a in valid:
            answer_counts[a] = answer_counts.get(a, 0) + 1
        most_common = max(answer_counts.items(), key=lambda x: x[1]) if answer_counts else ("", 0)

        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "sentence_idx": sentence_idx, "forced_prefix": forced_prefix,
            "total_resamples": len(resample_results), "valid_answers": len(valid),
            "answer_counts": answer_counts, "most_common": most_common[0],
            "most_common_count": most_common[1],
            "agreement_rate": most_common[1] / len(valid) if valid else 0,
        }
        path = sentence_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def _save_resampling_summary(self, question: Question, source_rollout_idx: int,
                                 all_sentence_results: List[Dict],
                                 run_dir: Optional[Path] = None) -> Path:
        save_dir = run_dir or (self.get_question_dir(question.id, "resampling") / f"rollout_{source_rollout_idx:03d}")
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "source_rollout_idx": source_rollout_idx,
            "num_sentences": len(all_sentence_results),
            "sentence_summaries": all_sentence_results,
        }
        if isinstance(question, GPQAQuestion):
            summary["correct_answer"] = question.correct_answer
        else:
            summary["bad_outcome"] = question.bad_outcome
        path = save_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_latest_verification_dir(self, question_id: str) -> Optional[Path]:
        qdir = self.verification_dir / question_id
        if not qdir.exists():
            return None
        timestamped = sorted(
            [d for d in qdir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'],
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
