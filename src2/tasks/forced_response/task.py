"""
ForcingTask — true prefill forcing via Tinker.

Standalone task (no shared base class beyond BaseTask). For each sentence
boundary in a source CoT, prefills the model's <think> block with the partial
CoT and lets it continue to produce an answer.
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
from ...utils.chat_template import build_thinking_prompt
from .prompts import get_cumulative_cot_segments


@dataclass
class ForceResult:
    """Result of a single true-forcing attempt."""
    sentence_idx: int
    force_idx: int
    partial_cot: str
    continued_cot: str
    raw_tokens: List[int]
    raw_response: str
    answer: str
    full_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx, "force_idx": self.force_idx,
            "partial_cot": self.partial_cot, "continued_cot": self.continued_cot,
            "raw_tokens": self.raw_tokens, "raw_response": self.raw_response,
            "answer": self.answer, "full_prompt": self.full_prompt,
        }


class ForcingTask(BaseTask):
    """
    Forcing task: sentence-by-sentence prefill via Tinker.

    run_data() forces a verified question's CoT at every sentence boundary,
    producing per-sentence answer distributions as ground truth.
    """

    def __init__(self, model: str,
                 data_dir: Optional[Path] = None):
        super().__init__("forcing", data_dir or (
            Path(__file__).parent.parent.parent / "data" / "forced_response"
        ))
        self.model = model

        # Sub-directories
        self.verification_dir = self.data_dir.parent / "verification_rollouts"
        self.forcing_dir = self.data_dir / "forcing"
        self.monitor_forcing_dir = self.data_dir / "monitor_forcing"

        for d in [self.verification_dir, self.forcing_dir, self.monitor_forcing_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(
        self,
        question_id: str,
        rollout_idx: int = 0,
        num_forces: int = 5,
        max_sentences: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        sentence_stride: int = 1,
        verbose: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Run true forcing for all sentences in the CoT via Tinker."""
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
        if max_sentences is not None:
            cot_segments = cot_segments[:max_sentences]

        # Apply stride: keep every Nth sentence (always include first and last)
        if sentence_stride > 1:
            all_indices = list(range(len(cot_segments)))
            kept = set(all_indices[::sentence_stride])
            kept.add(all_indices[-1])  # always keep last sentence
            cot_segments = [cot_segments[i] for i in sorted(kept)]

        num_sentences = len(cot_segments)

        config = {
            "model": self.model, "question_id": question.id,
            "rollout_idx": rollout_idx, "num_forces": num_forces,
            "max_sentences": max_sentences, "sentence_stride": sentence_stride,
            "temperature": temperature,
            "max_tokens": max_tokens, "num_sentences": num_sentences,
        }
        run_dir = self.create_run_dir("forcing", question.id, rollout_idx, config)

        if verbose:
            print(f"Forcing {question.id}: {num_sentences} sentences × {num_forces} forces")
            print(f"Run dir: {run_dir}")

        with contextlib.redirect_stdout(io.StringIO()):
            source_token_count = len(tokenizer.encode(source_cot, add_special_tokens=False))

        def force_single(sent_idx: int, force_idx: int, partial_cot: str) -> Optional[ForceResult]:
            remaining_frac = 1.0 - len(partial_cot) / max(len(source_cot), 1)
            sent_max_tokens = min(max(int(source_token_count * remaining_frac) + 100, 100), 1024)

            prompt_str = build_thinking_prompt(
                tokenizer, self._user_msg(question), cot_prefix=partial_cot,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            model_input = types.ModelInput.from_ints(tokens)

            params = types.SamplingParams(max_tokens=sent_max_tokens, temperature=temperature)
            result = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=params).result()

            sample_tokens = result.sequences[0].tokens
            answer, raw_response = self._extract_answer(sample_tokens, tokenizer, question)

            if answer:
                continued_cot = raw_response.split("</think>", 1)[0] if "</think>" in raw_response else raw_response
                return ForceResult(
                    sentence_idx=sent_idx, force_idx=force_idx,
                    partial_cot=partial_cot, continued_cot=continued_cot,
                    raw_tokens=list(sample_tokens), raw_response=raw_response, answer=answer,
                    full_prompt=prompt_str,
                )
            return None

        all_tasks = [
            (si, fi, cot_segments[si])
            for si in range(num_sentences) for fi in range(num_forces)
        ]
        all_results: List[ForceResult] = []

        with ThreadPoolExecutor(max_workers=min(300, len(all_tasks))) as executor:
            futures = {executor.submit(force_single, si, fi, pc): (si, fi) for si, fi, pc in all_tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Forcing", disable=not verbose):
                r = future.result()
                if r:
                    all_results.append(r)

        by_sentence: Dict[int, List[ForceResult]] = {}
        for r in all_results:
            by_sentence.setdefault(r.sentence_idx, []).append(r)

        all_summaries = []
        for si in range(num_sentences):
            sent_results = by_sentence.get(si, [])
            self._save_forcing_result(
                question=question, sentence_idx=si,
                partial_cot=cot_segments[si],
                force_results=[r.to_dict() for r in sent_results],
                run_dir=run_dir,
            )
            counts = {}
            for r in sent_results:
                counts[r.answer] = counts.get(r.answer, 0) + 1
            all_summaries.append({
                "sentence_idx": si, "partial_cot_length": len(cot_segments[si]),
                "total_forces": len(sent_results), "valid_answers": len(sent_results),
                "answer_counts": counts,
                "most_common": max(counts.items(), key=lambda x: x[1])[0] if counts else "",
            })

        self._save_forcing_summary(question, rollout_idx, all_summaries, source_cot, run_dir)

        if verbose:
            print(f"Done: {num_sentences} sentences processed.")

        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "num_sentences": num_sentences, "sentence_results": all_summaries,
        }
        if isinstance(question, GPQAQuestion):
            summary["correct_answer"] = question.correct_answer
        else:
            summary["bad_outcome"] = question.bad_outcome
        return summary

    def get_data(self, load: bool = False) -> Union[bool, Optional[Any]]:
        if not load:
            return self.forcing_dir.exists() and any(self.forcing_dir.rglob("summary.json"))
        summaries = sorted(self.forcing_dir.rglob("summary.json"))
        if not summaries:
            return None
        results = []
        for p in summaries:
            with open(p) as f:
                results.append(json.load(f))
        return results

    def get_activations(self, load: bool = False) -> Union[bool, Optional[List[Path]]]:
        if not load:
            return self.forcing_dir.exists() and any(self.forcing_dir.rglob("force_*.json"))
        paths = sorted(self.forcing_dir.rglob("force_*.json"))
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
                # Each sentence result has answer distributions
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
        # Load question definitions for question text and choices
        from ...utils.questions import load_custom_questions
        questions_file = Path(__file__).parent.parent.parent / "utils" / "questions.json"
        questions = load_custom_questions(questions_file)
        question_lookup = {q.id: q for q in questions}

        # Find all sentence-level summaries (which have the actual partial_cot)
        sentence_summaries = sorted(self.forcing_dir.rglob("sentence_*/summary.json"))
        sentence_summaries = data_slice.filter_paths(sentence_summaries)

        rows = []
        for summary_path in sentence_summaries:
            with open(summary_path) as f:
                sent = json.load(f)

            question_id = sent.get("question_id", "")
            if not data_slice.matches_id(question_id):
                continue

            sentence_idx = sent.get("sentence_idx", -1)
            if not data_slice.matches_sentence(sentence_idx):
                continue

            # Get question details from lookup
            question = question_lookup.get(question_id)
            if question is None:
                continue

            row = {
                "question_id": question_id,
                "question_type": sent.get("question_type", "multiple_choice"),
                "partial_cot": sent.get("partial_cot", ""),
                "sentence_idx": sentence_idx,
                "question": question.question,
            }

            # Add choices for multiple choice questions
            if hasattr(question, "choices"):
                row["choices"] = question.choices

            if hasattr(question, "correct_answer"):
                row["correct_answer"] = question.correct_answer
            if hasattr(question, "bad_outcome"):
                row["bad_outcome"] = question.bad_outcome

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
        Extract and save full-sequence activations for forcing runs.

        Walks all force_*.json files, extracts full [seq_len, hidden_dim]
        activations plus token boundary indices, and saves them as
        companion .npz files.
        """
        from ...utils.activations import ActivationExtractor, compute_token_boundaries

        if not self.forcing_dir.exists():
            raise RuntimeError(f"No forcing data found at {self.forcing_dir}")

        run_files = sorted(self.forcing_dir.rglob("force_*.json"))
        if not run_files:
            raise RuntimeError(f"No force run files found in {self.forcing_dir}")

        run_files = data_slice.filter_paths(run_files)

        filtered_files = []
        for rp in run_files:
            # Parse question_id and sentence_idx from path
            parts = rp.parts
            question_id = None
            sentence_idx = None
            for part in parts:
                if part.startswith("sentence_"):
                    try:
                        sentence_idx = int(part.split("_", 1)[1])
                    except ValueError:
                        pass
            # question_id is the directory after "forcing"
            try:
                forcing_idx = parts.index("forcing")
                if forcing_idx + 1 < len(parts):
                    question_id = parts[forcing_idx + 1]
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

        for run_path in tqdm(run_files, desc="Extracting activations (forcing)"):
            act_path = run_path.with_suffix(".npz")
            if act_path.exists():
                with np.load(act_path) as f:
                    if seq_key in f.files:
                        continue

            with open(run_path) as f:
                run_data = json.load(f)

            raw_response = run_data.get("raw_response", "")
            if not raw_response:
                continue

            full_prompt = run_data.get("full_prompt", "")
            if not full_prompt:
                import logging
                logging.warning(
                    "No full_prompt in %s — skipping (old data without full context)",
                    run_path,
                )
                continue

            full_text = full_prompt + raw_response

            try:
                act = extractor.extract_full_sequence(full_text, layer)
                boundaries = compute_token_boundaries(
                    extractor.tokenizer, full_prompt, raw_response,
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

        print(f"Activations extracted for forcing data in {self.forcing_dir}")

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
                "partial_cot": str,
                "continued_cot": str,
                "raw_response": str,
                "full_prompt": str,
                "answer": str,
            }
        """
        seq_key = f"layer{layer}_full_sequence"
        bnd_key = f"layer{layer}_boundaries"
        legacy_key = f"layer{layer}_{token_position}"
        boundary_names = ["last_input", "last_thinking", "last_response"]

        # Load sentence-level summaries to get answer distributions
        summary_files = sorted(self.forcing_dir.rglob("sentence_*/summary.json"))
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
            question_type = summary.get("question_type", "multiple_choice")

            # Find companion activation files
            act_files = sorted(sentence_dir.glob("force_*.npz"))
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
                    "question_type": question_type,
                    "sentence_idx": sentence_idx,
                    "partial_cot": text_data.get("partial_cot", ""),
                    "continued_cot": text_data.get("continued_cot", ""),
                    "raw_response": text_data.get("raw_response", ""),
                    "full_prompt": text_data.get("full_prompt", ""),
                    "answer": text_data.get("answer", ""),
                })

        return samples

    # ------------------------------------------------------------------
    # Attention probe data builder
    # ------------------------------------------------------------------

    # Label maps for encoding forced answers as integer classes
    GPQA_LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
    BINARY_LABEL_MAP = {"YES": 0, "NO": 1}

    def build_attention_probe_data(
        self,
        layer: int,
        data_slice: DataSlice,
        token_position: str = "full_sequence",
    ) -> Dict[str, Any]:
        """
        Build data formatted for AttentionProbe in classification mode.

        Each sample maps a forced-response activation to a hard class label
        (the answer produced by that forcing run).

        Args:
            layer: Model layer to load activations from.
            data_slice: Filter for question IDs and sentence indices.
            token_position: Token position mode (default "full_sequence").

        Returns:
            {
                "X_list": List[ndarray],   # activations
                "y": ndarray[int],         # integer class labels
                "question_ids": list,
                "sentence_indices": list,
            }
        """
        samples = self.get_probe_data(layer, data_slice, token_position)

        X_list = []
        y_list = []
        question_ids = []
        sentence_indices = []

        for sample in samples:
            answer = sample.get("answer", "").upper()
            if not answer:
                continue

            # Determine label map from question type
            question_type = sample.get("question_type", "")
            if not question_type:
                # Infer from answer value
                if answer in self.GPQA_LABEL_MAP:
                    label_map = self.GPQA_LABEL_MAP
                elif answer in self.BINARY_LABEL_MAP:
                    label_map = self.BINARY_LABEL_MAP
                else:
                    continue
            elif question_type == "binary_judge":
                label_map = self.BINARY_LABEL_MAP
            else:
                label_map = self.GPQA_LABEL_MAP

            if answer not in label_map:
                continue

            X_list.append(sample["activation"])
            y_list.append(label_map[answer])
            question_ids.append(sample["question_id"])
            sentence_indices.append(sample["sentence_idx"])

        return {
            "X_list": X_list,
            "y": np.array(y_list, dtype=np.int64),
            "question_ids": question_ids,
            "sentence_indices": sentence_indices,
        }

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
            "forcing": self.forcing_dir,
            "monitor_forcing": self.monitor_forcing_dir,
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

    def _save_forcing_result(self, question: Question, sentence_idx: int,
                             partial_cot: str, force_results: List[Dict],
                             run_dir: Path) -> Path:
        sentence_dir = run_dir / f"sentence_{sentence_idx:03d}"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(force_results):
            with open(sentence_dir / f"force_{i:03d}.json", "w") as f:
                json.dump(result, f, indent=2)

        summary = self._build_sentence_summary(question, sentence_idx, partial_cot, force_results)
        summary_path = sentence_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    def _save_forcing_summary(self, question: Question, source_rollout_idx: int,
                              all_sentence_results: List[Dict], source_cot: str = "",
                              run_dir: Optional[Path] = None) -> Path:
        save_dir = run_dir or (self.get_question_dir(question.id, "forcing") / f"rollout_{source_rollout_idx:03d}")
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "question_id": question.id, "question_type": question.question_type,
            "source_rollout_idx": source_rollout_idx,
            "num_sentences": len(all_sentence_results),
            "source_cot": source_cot, "sentence_summaries": all_sentence_results,
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
    def _build_sentence_summary(question: Question, sentence_idx: int,
                                partial_cot: str, results: List[Dict]) -> Dict:
        answers = [r.get("answer", "").upper() for r in results]
        if isinstance(question, BinaryJudgeQuestion):
            valid = [a for a in answers if a in ["YES", "NO"]]
        else:
            valid = [a for a in answers if a in ["A", "B", "C", "D"]]
        counts = {}
        for a in valid:
            counts[a] = counts.get(a, 0) + 1
        return {
            "question_id": question.id, "question_type": question.question_type,
            "sentence_idx": sentence_idx, "partial_cot": partial_cot,
            "total_attempts": len(results), "valid_answers": len(valid),
            "answer_counts": counts,
        }

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
        if isinstance(question, BinaryJudgeQuestion):
            if "</think>" in text:
                after = text.split("</think>", 1)[1].strip().upper()
                if "YES" in after:
                    return "YES", text
                if "NO" in after:
                    return "NO", text
            return "", text
        if "</think>" in text:
            after = text.split("</think>", 1)[1].strip().upper().rstrip(".")
            if after in ["A", "B", "C", "D"]:
                return after, text
            match = re.search(r"\b([A-D])\b", after)
            if match:
                return match.group(1), text
        return "", text
