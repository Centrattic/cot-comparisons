"""
Thought Anchors — pure adaptive sentence importance scoring via resampling.

For each sentence in the middle section of a CoT, measures importance by
removing it and running continuations to see how the answer distribution
changes. Uses an adaptive sampling schedule with bootstrap confidence
intervals for early stopping (pure adaptive strategy from Frugal Thought
Anchors).

After scoring all sentences, selects the top-K most important ones to
produce a compressed CoT.
"""

import contextlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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


class ThoughtAnchors(BaseMethod):
    """
    Pure adaptive thought anchor identification for CoT compression.

    For each sentence in the middle section, adaptively samples continuations
    with that sentence removed. Compares answer distributions to baseline
    (full CoT) to compute importance. Stops sampling early when a bootstrap
    CI decisively classifies a sentence as IMPORTANT or NEGLIGIBLE.

    Usage:
        anchors = ThoughtAnchors(model="moonshotai/kimi-k2-thinking")
        anchors.set_task(task)
        folder = anchors.get_folder()
        results = anchors.infer(task.prepare_for_thought_anchors())
        anchors._output.mark_success()
    """

    def __init__(
        self,
        model: str,
        sampling_schedule: Optional[List[int]] = None,
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
        2. For each middle sentence, adaptively measure importance
        3. Select top-K sentences, produce compressed CoT

        Args:
            data: List of row dicts from task.prepare_for_thought_anchors().
            verbose: Show progress bars.

        Returns:
            List of result dicts with importance scores and compressed CoTs.
        """
        from tinker import ServiceClient, types
        from transformers import AutoTokenizer

        if self._output is None or self._output.run_folder is None:
            raise RuntimeError("Call set_task() and get_folder() before infer().")

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
        """Process a single compression spec: baseline + per-sentence importance."""
        sentences = row["sentences"]
        middle_start = row.get("middle_start_idx", row.get("middle_start", 0))
        middle_end = row.get("middle_end_idx", row.get("middle_end", len(sentences)))
        target_n = row.get("target_num_sentences", 5)
        char_budget = row.get("char_budget", 1000)
        question_text = row.get("question", "")
        question_type = row.get("question_type", "multiple_choice")

        # Build the question message for Tinker prompts
        if question_type == "binary_judge":
            user_msg = question_text
        else:
            choices = row.get("choices", [])
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            choices_text = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
            user_msg = f"{question_text}\n\n{choices_text}\n\nAnswer with just the letter (A, B, C, or D)."

        full_cot = " ".join(sentences)

        # Step 1: Baseline — run continuations from full CoT
        if verbose:
            print(f"  Computing baseline ({self.sampling_schedule[0]} samples)...")
        baseline_answers = self._run_continuations(
            user_msg, full_cot, self.sampling_schedule[0],
            tokenizer, sampling_client, params, types, question_type,
        )
        baseline_correct = self._most_common_answer(baseline_answers)

        # Step 2: Per-sentence importance scoring (adaptive)
        middle_indices = list(range(middle_start, middle_end))
        importance_scores = {}
        sentence_classifications = {}
        total_samples_used = 0

        desc = "  Scoring sentences" if verbose else None
        iterator = tqdm(middle_indices, desc=desc, disable=not verbose)

        for sent_idx in iterator:
            # Build CoT with this sentence removed
            modified_sentences = sentences[:sent_idx] + sentences[sent_idx + 1:]
            modified_cot = " ".join(modified_sentences)

            importance, classification, n_samples = self._adaptive_importance(
                user_msg, modified_cot, baseline_correct, baseline_answers,
                tokenizer, sampling_client, params, types, question_type,
            )
            importance_scores[sent_idx] = importance
            sentence_classifications[sent_idx] = classification
            total_samples_used += n_samples

        # Step 3: Select top-K most important sentences
        ranked = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True,
        )

        # Greedy selection: pick by importance until we hit target count or char budget
        selected_indices = []
        selected_chars = 0
        for sent_idx, score in ranked:
            if len(selected_indices) >= target_n:
                break
            sent_len = len(sentences[sent_idx])
            if selected_chars + sent_len <= char_budget:
                selected_indices.append(sent_idx)
                selected_chars += sent_len

        # If we haven't filled target_n, allow exceeding char budget slightly
        if len(selected_indices) < target_n:
            for sent_idx, score in ranked:
                if sent_idx in selected_indices:
                    continue
                if len(selected_indices) >= target_n:
                    break
                selected_indices.append(sent_idx)

        selected_indices.sort()

        # Reconstruct compressed CoT
        middle_sentences = [sentences[i] for i in range(middle_start, middle_end)]
        # Convert absolute indices to relative middle indices for reconstruction
        relative_selected = [i - middle_start for i in selected_indices]
        compressed_middle = " ".join(
            middle_sentences[i] for i in sorted(relative_selected) if 0 <= i < len(middle_sentences)
        )

        first_quarter = " ".join(sentences[:middle_start])
        last_quarter = " ".join(sentences[middle_end:])
        parts = [p for p in [first_quarter, compressed_middle, last_quarter] if p]
        compressed_cot = " ".join(parts)

        return {
            "question_id": row.get("question_id", ""),
            "rollout_idx": row.get("rollout_idx", 0),
            "importance_scores": {str(k): v for k, v in importance_scores.items()},
            "sentence_classifications": {str(k): v for k, v in sentence_classifications.items()},
            "selected_indices": selected_indices,
            "compressed_middle": compressed_middle,
            "compressed_cot": compressed_cot,
            "target_num_sentences": target_n,
            "char_budget": char_budget,
            "actual_num_selected": len(selected_indices),
            "actual_char_length": len(compressed_middle),
            "total_samples_used": total_samples_used,
            "baseline_answer": baseline_correct,
            "num_middle_sentences": middle_end - middle_start,
        }

    def _adaptive_importance(
        self,
        user_msg: str,
        modified_cot: str,
        baseline_answer: str,
        baseline_answers: List[str],
        tokenizer, sampling_client, params, types,
        question_type: str,
    ) -> Tuple[float, str, int]:
        """
        Adaptively measure importance of a sentence removal.

        Returns (importance_score, classification, num_samples_used).
        classification is one of: "important", "negligible", "uncertain".
        """
        all_answers = []
        prev_checkpoint = 0

        for checkpoint in self.sampling_schedule:
            # Run incremental samples
            n_new = checkpoint - prev_checkpoint
            new_answers = self._run_continuations(
                user_msg, modified_cot, n_new,
                tokenizer, sampling_client, params, types, question_type,
            )
            all_answers.extend(new_answers)
            prev_checkpoint = checkpoint

            # Compute importance: fraction of baseline answers matching baseline_answer
            # minus fraction of modified answers matching baseline_answer
            baseline_rate = sum(
                1 for a in baseline_answers if a == baseline_answer
            ) / len(baseline_answers) if baseline_answers else 0

            modified_rate = sum(
                1 for a in all_answers if a == baseline_answer
            ) / len(all_answers) if all_answers else 0

            importance = baseline_rate - modified_rate

            # Bootstrap CI
            ci_low, ci_high = self._bootstrap_ci(
                baseline_answers, all_answers, baseline_answer,
            )

            # Classification
            if ci_low > 0:
                return importance, "important", checkpoint
            if ci_high < 0:
                return importance, "negligible", checkpoint

        # Exhausted schedule — return best estimate
        return importance, "uncertain", self.sampling_schedule[-1]

    def _bootstrap_ci(
        self,
        baseline_answers: List[str],
        modified_answers: List[str],
        target_answer: str,
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for importance = baseline_rate - modified_rate.
        """
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

    def _run_continuations(
        self,
        user_msg: str,
        cot_prefix: str,
        n: int,
        tokenizer, sampling_client, params, types,
        question_type: str,
    ) -> List[str]:
        """Run n continuations from a CoT prefix via Tinker, return answers."""
        prompt_str = (
            f"{IM_SYSTEM}system{IM_MIDDLE}You are Kimi, an AI assistant created by Moonshot AI.{IM_END}"
            f"{IM_USER}user{IM_MIDDLE}{user_msg}{IM_END}"
            f"{IM_ASSISTANT}assistant{IM_MIDDLE}<think>{cot_prefix}"
        )

        def run_single(_idx: int) -> Optional[str]:
            with contextlib.redirect_stdout(io.StringIO()):
                tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
            model_input = types.ModelInput.from_ints(tokens)
            result = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=params,
            ).result()
            sample_tokens = result.sequences[0].tokens
            return self._extract_answer(sample_tokens, tokenizer, question_type)

        answers = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as executor:
            futures = {executor.submit(run_single, i): i for i in range(n)}
            for future in as_completed(futures):
                try:
                    ans = future.result()
                    if ans:
                        answers.append(ans)
                except Exception:
                    pass
        return answers

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
        """Return the most common answer."""
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
            "confidence_level": self.confidence_level,
            "bootstrap_reps": self.bootstrap_reps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_rows": len(results),
        }
        with open(folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Summary CSV
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
                "num_middle_sentences": r.get("num_middle_sentences", 0),
                "num_selected": r.get("actual_num_selected", 0),
                "compressed_chars": r.get("actual_char_length", 0),
                "char_budget": r.get("char_budget", 0),
                "total_samples": r.get("total_samples_used", 0),
                "n_important": n_important,
                "n_negligible": n_negligible,
                "n_uncertain": n_uncertain,
                "baseline_answer": r.get("baseline_answer", ""),
            })

        if summary_rows:
            import pandas as pd
            pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False)
