"""
Thought Anchors — hybrid sentence importance scoring via resampling.

For each sentence in the compression region, measures importance by removing
it and running continuations to see how the answer distribution changes.

Uses the hybrid strategy from Frugal Thought Anchors:
  1. Sparse pass: sample every 3rd sentence with adaptive early stopping
  2. Fill-in: where accuracy jumps ≥10% between sampled neighbors, fill gaps
  3. Interpolate: remaining sentences inherit from neighbors (conservative)

All Tinker calls within each pass are launched in parallel across sentences.
"""

import contextlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .base import BaseMethod

# Kimi K2 chat template tokens
IM_SYSTEM = "<|im_system|>"
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"


@dataclass
class SentenceResult:
    """Result for a single sentence's importance analysis."""
    sent_idx: int
    accuracy: float  # accuracy when this sentence is removed
    importance: float  # baseline_rate - modified_rate
    classification: str  # "important", "negligible", "uncertain", "interpolated"
    samples_used: int
    source: str  # "sparse", "fillin", "interpolated"
    ci: Tuple[float, float] = (0.0, 0.0)


class ThoughtAnchors(BaseMethod):
    """
    Hybrid thought anchor identification for CoT compression.

    Scores sentence importance by removing each sentence and running
    continuations. Uses a 3-pass hybrid strategy (sparse → fill-in →
    interpolate) with all Tinker calls parallelized across sentences.

    Usage:
        anchors = ThoughtAnchors(model="Qwen/Qwen3-32B")
        anchors.set_task(task)
        results = anchors.infer(task.prepare_for_thought_anchors())
        anchors._output.mark_success()
    """

    def __init__(
        self,
        model: str,
        sampling_schedule: Optional[List[int]] = None,
        sparse_step: int = 3,
        jump_threshold: float = 0.10,
        confidence_level: float = 0.95,
        bootstrap_reps: int = 1000,
        max_workers: int = 300,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        name: Optional[str] = None,
    ):
        super().__init__(name or "thought_anchors")
        self.model = model
        self.sampling_schedule = sampling_schedule or [10, 20, 30, 50, 70, 100]
        self.sparse_step = sparse_step
        self.jump_threshold = jump_threshold
        self.confidence_level = confidence_level
        self.bootstrap_reps = bootstrap_reps
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens

    def infer(self, data: List[Dict], verbose: bool = True) -> List[Dict]:
        """
        Run thought anchor analysis on each compression spec.

        For each row:
        1. Compute baseline answer distribution from full CoT
        2. Hybrid 3-pass importance scoring (all sentences in parallel)
        3. Select top-K sentences, produce compressed CoT
        """
        from tinker import ServiceClient, types
        from transformers import AutoTokenizer

        if self._output is None:
            raise RuntimeError("Call set_task() before infer().")

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        client = ServiceClient()
        sampling_client = client.create_sampling_client(base_model=self.model)
        params = types.SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature,
        )

        all_results = []

        for row_idx, row in enumerate(data):
            if verbose:
                print(f"\nProcessing row {row_idx + 1}/{len(data)}: {row.get('question_id', '?')}")

            result = self._process_row(
                row, tokenizer, sampling_client, params, types, verbose,
            )
            all_results.append(result)

        self._save_results(all_results)
        return all_results

    def _process_row(
        self, row: Dict, tokenizer, sampling_client, params, types, verbose: bool,
    ) -> Dict:
        """Process a single compression spec using hybrid 3-pass strategy."""
        sentences = row["sentences"]
        region_start = row.get("region_start_idx", row.get("middle_start_idx", 0))
        region_end = row.get("region_end_idx", row.get("middle_end_idx", len(sentences)))
        target_n = row.get("target_num_sentences", 5)
        char_budget = row.get("char_budget", 1000)
        question_text = row.get("question", "")
        question_type = row.get("question_type", "multiple_choice")

        if question_type == "binary_judge":
            user_msg = question_text
        else:
            choices = row.get("choices", [])
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            choices_text = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
            user_msg = f"{question_text}\n\n{choices_text}\n\nAnswer with just the letter (A, B, C, or D)."

        full_cot = " ".join(sentences)
        region_indices = list(range(region_start, region_end))

        # Pre-tokenize the prompt template (shared across all continuations)
        prompt_template = (
            f"{IM_SYSTEM}system{IM_MIDDLE}You are Kimi, an AI assistant created by Moonshot AI.{IM_END}"
            f"{IM_USER}user{IM_MIDDLE}{user_msg}{IM_END}"
            f"{IM_ASSISTANT}assistant{IM_MIDDLE}<think>"
        )

        # Build all modified CoTs upfront (sentence i removed)
        modified_cots = {}
        for idx in region_indices:
            modified = sentences[:idx] + sentences[idx + 1:]
            modified_cots[idx] = " ".join(modified)

        # Step 1: Baseline — run continuations from full CoT
        n_baseline = self.sampling_schedule[0]
        if verbose:
            print(f"  Computing baseline ({n_baseline} samples)...")
        baseline_answers = self._run_continuations_batch(
            prompt_template, {-1: full_cot}, n_baseline,
            tokenizer, sampling_client, params, types, question_type,
        )[-1]
        baseline_answer = self._most_common_answer(baseline_answers)
        baseline_rate = (
            sum(1 for a in baseline_answers if a == baseline_answer) / len(baseline_answers)
            if baseline_answers else 0
        )

        # ====================================================================
        # Pass 1: Sparse — sample every sparse_step-th sentence, adaptive
        # ====================================================================
        sparse_indices = [idx for i, idx in enumerate(region_indices) if i % self.sparse_step == 0]
        if verbose:
            print(f"  Pass 1: sparse sampling {len(sparse_indices)}/{len(region_indices)} sentences...")

        results: Dict[int, SentenceResult] = {}
        sparse_cots = {idx: modified_cots[idx] for idx in sparse_indices}
        self._adaptive_batch(
            prompt_template, sparse_cots, baseline_answer, baseline_rate, baseline_answers,
            tokenizer, sampling_client, params, types, question_type,
            results, source="sparse", verbose=verbose,
        )

        # ====================================================================
        # Pass 2: Fill-in — where accuracy jumps between sparse neighbors
        # ====================================================================
        fillin_indices = set()
        for i in range(len(sparse_indices) - 1):
            idx_a = sparse_indices[i]
            idx_b = sparse_indices[i + 1]
            acc_a = results[idx_a].accuracy
            acc_b = results[idx_b].accuracy
            if abs(acc_b - acc_a) >= self.jump_threshold:
                for idx in region_indices:
                    if idx_a < idx < idx_b and idx not in results:
                        fillin_indices.add(idx)

        if fillin_indices:
            if verbose:
                print(f"  Pass 2: fill-in {len(fillin_indices)} sentences...")
            fillin_cots = {idx: modified_cots[idx] for idx in fillin_indices}
            self._adaptive_batch(
                prompt_template, fillin_cots, baseline_answer, baseline_rate, baseline_answers,
                tokenizer, sampling_client, params, types, question_type,
                results, source="fillin", verbose=verbose,
            )

        # ====================================================================
        # Pass 3: Interpolate remaining from neighbors
        # ====================================================================
        for idx in region_indices:
            if idx not in results:
                acc, classification = self._interpolate(idx, results, baseline_rate)
                results[idx] = SentenceResult(
                    sent_idx=idx, accuracy=acc,
                    importance=baseline_rate - acc,
                    classification=classification,
                    samples_used=0, source="interpolated",
                )

        n_interp = sum(1 for r in results.values() if r.source == "interpolated")
        if verbose and n_interp:
            print(f"  Pass 3: interpolated {n_interp} sentences")

        total_samples = n_baseline + sum(r.samples_used for r in results.values())

        # ====================================================================
        # Select top-K and reconstruct
        # ====================================================================
        ranked = sorted(results.values(), key=lambda r: r.importance, reverse=True)

        selected_indices = []
        selected_chars = 0
        for r in ranked:
            if len(selected_indices) >= target_n:
                break
            sent_len = len(sentences[r.sent_idx])
            if selected_chars + sent_len <= char_budget:
                selected_indices.append(r.sent_idx)
                selected_chars += sent_len

        if len(selected_indices) < target_n:
            for r in ranked:
                if r.sent_idx in selected_indices:
                    continue
                if len(selected_indices) >= target_n:
                    break
                selected_indices.append(r.sent_idx)

        selected_indices.sort()

        region_sentences = sentences[region_start:region_end]
        relative_selected = [i - region_start for i in selected_indices]
        compressed_region = " ".join(
            region_sentences[i] for i in sorted(relative_selected) if 0 <= i < len(region_sentences)
        )

        pre_region = " ".join(sentences[:region_start])
        post_region = " ".join(sentences[region_end:])
        parts = [p for p in [pre_region, compressed_region, post_region] if p]
        compressed_cot = " ".join(parts)

        importance_scores = {str(r.sent_idx): r.importance for r in results.values()}
        classifications = {str(r.sent_idx): r.classification for r in results.values()}

        return {
            "question_id": row.get("question_id", ""),
            "rollout_idx": row.get("rollout_idx", 0),
            "importance_scores": importance_scores,
            "sentence_classifications": classifications,
            "selected_indices": selected_indices,
            "compressed_region": compressed_region,
            "compressed_cot": compressed_cot,
            "target_num_sentences": target_n,
            "char_budget": char_budget,
            "actual_num_selected": len(selected_indices),
            "actual_char_length": len(compressed_region),
            "total_samples_used": total_samples,
            "baseline_answer": baseline_answer,
            "num_region_sentences": region_end - region_start,
            "sparse_count": sum(1 for r in results.values() if r.source == "sparse"),
            "fillin_count": sum(1 for r in results.values() if r.source == "fillin"),
            "interpolated_count": n_interp,
        }

    # ------------------------------------------------------------------
    # Hybrid helpers
    # ------------------------------------------------------------------

    def _adaptive_batch(
        self,
        prompt_template: str,
        cots: Dict[int, str],  # sent_idx -> modified CoT
        baseline_answer: str,
        baseline_rate: float,
        baseline_answers: List[str],
        tokenizer, sampling_client, params, types,
        question_type: str,
        results: Dict[int, SentenceResult],
        source: str,
        verbose: bool,
    ) -> None:
        """
        Run adaptive importance scoring for a batch of sentences in parallel.

        For each checkpoint in the schedule, launches all unresolved sentences'
        continuations in parallel, then checks CIs to classify what we can.
        """
        # Track accumulated answers per sentence
        answers_per_sent: Dict[int, List[str]] = {idx: [] for idx in cots}
        unresolved = set(cots.keys())
        prev_checkpoint = 0

        for checkpoint in self.sampling_schedule:
            if not unresolved:
                break

            n_new = checkpoint - prev_checkpoint
            prev_checkpoint = checkpoint

            # Run continuations for ALL unresolved sentences in parallel
            batch_cots = {idx: cots[idx] for idx in unresolved}
            batch_answers = self._run_continuations_batch(
                prompt_template, batch_cots, n_new,
                tokenizer, sampling_client, params, types, question_type,
            )

            newly_resolved = []
            for idx in unresolved:
                answers_per_sent[idx].extend(batch_answers.get(idx, []))
                all_answers = answers_per_sent[idx]

                if not all_answers:
                    continue

                modified_rate = sum(1 for a in all_answers if a == baseline_answer) / len(all_answers)
                importance = baseline_rate - modified_rate

                ci_low, ci_high = self._bootstrap_ci(
                    baseline_answers, all_answers, baseline_answer,
                )

                if ci_low > 0:
                    classification = "important"
                elif ci_high < 0:
                    classification = "negligible"
                else:
                    classification = "uncertain"

                if classification != "uncertain" or checkpoint == self.sampling_schedule[-1]:
                    results[idx] = SentenceResult(
                        sent_idx=idx, accuracy=modified_rate,
                        importance=importance, classification=classification,
                        samples_used=len(all_answers), source=source,
                        ci=(ci_low, ci_high),
                    )
                    newly_resolved.append(idx)

            unresolved -= set(newly_resolved)

            if verbose:
                n_total = len(cots)
                n_done = n_total - len(unresolved)
                print(f"    checkpoint {checkpoint}: {n_done}/{n_total} resolved")

    def _run_continuations_batch(
        self,
        prompt_template: str,
        cots: Dict[int, str],  # key -> CoT text
        n_per_cot: int,
        tokenizer, sampling_client, params, types,
        question_type: str,
    ) -> Dict[int, List[str]]:
        """
        Run n_per_cot continuations for each CoT in parallel.

        Returns dict mapping key -> list of answer strings.
        """
        # Pre-tokenize all prompts
        tokenized = {}
        for key, cot in cots.items():
            prompt_str = prompt_template + cot
            with contextlib.redirect_stdout(io.StringIO()):
                tokenized[key] = tokenizer.encode(prompt_str, add_special_tokens=False)

        # Build all (key, sample_idx) jobs
        jobs = []
        for key in cots:
            for i in range(n_per_cot):
                jobs.append((key, i))

        answers: Dict[int, List[str]] = {key: [] for key in cots}

        def run_single(key_and_idx):
            key, _ = key_and_idx
            tokens = tokenized[key]
            model_input = types.ModelInput.from_ints(tokens)
            result = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=params,
            ).result()
            sample_tokens = result.sequences[0].tokens
            return key, self._extract_answer(sample_tokens, tokenizer, question_type)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_single, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    key, ans = future.result()
                    if ans:
                        answers[key].append(ans)
                except Exception:
                    pass

        return answers

    @staticmethod
    def _interpolate(
        idx: int,
        results: Dict[int, SentenceResult],
        baseline_rate: float,
    ) -> Tuple[float, str]:
        """Conservative interpolation from nearest sampled neighbors."""
        below = [(r.sent_idx, r) for r in results.values() if r.sent_idx < idx]
        above = [(r.sent_idx, r) for r in results.values() if r.sent_idx > idx]

        prev = max(below, key=lambda x: x[0]) if below else None
        nxt = min(above, key=lambda x: x[0]) if above else None

        if prev and nxt:
            _, prev_r = prev
            _, nxt_r = nxt
            # Use nearer neighbor's accuracy
            if idx - prev_r.sent_idx <= nxt_r.sent_idx - idx:
                acc = prev_r.accuracy
            else:
                acc = nxt_r.accuracy
            # Conservative: if either neighbor is important, mark important
            if prev_r.classification == "important" or nxt_r.classification == "important":
                return acc, "important"
            if prev_r.classification == nxt_r.classification:
                return acc, prev_r.classification
            return acc, "uncertain"
        elif prev:
            return prev[1].accuracy, prev[1].classification
        elif nxt:
            return nxt[1].accuracy, nxt[1].classification
        else:
            return baseline_rate, "uncertain"

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self,
        baseline_answers: List[str],
        modified_answers: List[str],
        target_answer: str,
    ) -> Tuple[float, float]:
        """Bootstrap CI for importance = baseline_rate - modified_rate."""
        rng = np.random.default_rng()
        baseline_arr = np.array([1 if a == target_answer else 0 for a in baseline_answers])
        modified_arr = np.array([1 if a == target_answer else 0 for a in modified_answers])

        n_base = len(baseline_arr)
        n_mod = len(modified_arr)
        if n_base == 0 or n_mod == 0:
            return -1.0, 1.0

        diffs = np.empty(self.bootstrap_reps)
        for i in range(self.bootstrap_reps):
            boot_base = rng.choice(baseline_arr, size=n_base, replace=True)
            boot_mod = rng.choice(modified_arr, size=n_mod, replace=True)
            diffs[i] = boot_base.mean() - boot_mod.mean()

        alpha = 1.0 - self.confidence_level
        ci_low = float(np.percentile(diffs, 100 * alpha / 2))
        ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    @staticmethod
    def _extract_answer(tokens, tokenizer, question_type: str) -> Optional[str]:
        """Extract answer from generated tokens."""
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if "</think>" in text:
            answer_text = text.split("</think>", 1)[1].strip()
        else:
            answer_text = ""

        if question_type == "binary_judge":
            upper = answer_text.upper()
            if "YES" in upper:
                return "YES"
            if "NO" in upper:
                return "NO"
            return None

        clean = answer_text.upper().rstrip(".").strip()
        if clean in ["A", "B", "C", "D"]:
            return clean
        match = re.search(r"\b([A-D])\b", answer_text)
        if match:
            return match.group(1).upper()
        return None

    @staticmethod
    def _most_common_answer(answers: List[str]) -> str:
        if not answers:
            return ""
        counts = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)

    def _save_results(self, results: List[Dict]) -> None:
        """Save results to the run folder."""
        folder = self._output.run_folder

        jsonl_path = folder / "results.jsonl"
        with open(jsonl_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        config = {
            "model": self.model,
            "sampling_schedule": self.sampling_schedule,
            "sparse_step": self.sparse_step,
            "jump_threshold": self.jump_threshold,
            "confidence_level": self.confidence_level,
            "bootstrap_reps": self.bootstrap_reps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_rows": len(results),
        }
        with open(folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        summary_rows = []
        for r in results:
            n_important = sum(
                1 for v in r.get("sentence_classifications", {}).values()
                if v == "important"
            )
            n_negligible = sum(
                1 for v in r.get("sentence_classifications", {}).values()
                if v == "negligible"
            )
            n_uncertain = sum(
                1 for v in r.get("sentence_classifications", {}).values()
                if v == "uncertain"
            )
            summary_rows.append({
                "question_id": r.get("question_id", ""),
                "num_region_sentences": r.get("num_region_sentences", 0),
                "num_selected": r.get("actual_num_selected", 0),
                "compressed_chars": r.get("actual_char_length", 0),
                "char_budget": r.get("char_budget", 0),
                "total_samples": r.get("total_samples_used", 0),
                "sparse_count": r.get("sparse_count", 0),
                "fillin_count": r.get("fillin_count", 0),
                "interpolated_count": r.get("interpolated_count", 0),
                "n_important": n_important,
                "n_negligible": n_negligible,
                "n_uncertain": n_uncertain,
                "baseline_answer": r.get("baseline_answer", ""),
            })

        if summary_rows:
            import pandas as pd
            pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False)
