"""
Run CoT Compression pipeline: BB Monitor, Faithful BB, and Last-N baseline
across all questions (custom + GPQA diamond).

Uses vLLM for local Qwen3-32B inference instead of Tinker.
GPT-5.2 monitor calls (via OpenRouter) stay the same.

Three phases per question (batched for speed):
  1. Prepare: generate compression specs + run monitors for all rollouts
  2. Batch evaluate: fire ALL resamples across ALL rollouts×methods via vLLM
  3. Save: compute KL, save per-rollout and aggregate results

Skips question×method combos that already have >= NUM_ROLLOUTS results.

Usage:
    python -m src2.runs.run_compression_vllm
    python -m src2.runs.run_compression_vllm --skip-bb
    python -m src2.runs.run_compression_vllm --skip-faithful
    python -m src2.runs.run_compression_vllm --skip-baseline
"""

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams as VllmSamplingParams

from src2.methods import LlmMonitor
from src2.tasks import CompressedCotTask
from src2.tasks.compressed_cot.task import CompressionSpec
from src2.tasks.compressed_cot.prompts import (
    FaithfulSentenceSelectionPrompt,
    SentenceSelectionPrompt,
)
from src2.methods.base import BaseMethod
from src2.utils.questions import GPQAQuestion
from src2.utils.chat_template import build_thinking_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SUBJECT_MODEL = "Qwen/Qwen3-32B"
MONITOR_MODEL = "openai/gpt-5.2"

COMPRESSION_FACTOR = 10
CHAR_LIMIT_MULTIPLIER = 1.5
COMPRESS_PCT = 0.5
REGION = "prefix"

EVAL_MODE = "kl_divergence"
NUM_ROLLOUTS = 35
NUM_RESAMPLES = 50

QUESTION_IDS = [
    # Custom questions
    "starfish",
    "custom_bagel_001",
    "waffle",
    # GPQA diamond (10 questions, high agreement)
    "gpqa_gpqa_diamond_0007",
    "gpqa_gpqa_diamond_0010",
    "gpqa_gpqa_diamond_0013",
    "gpqa_gpqa_diamond_0019",
    "gpqa_gpqa_diamond_0026",
    "gpqa_gpqa_diamond_0031",
    "gpqa_gpqa_diamond_0034",
    "gpqa_gpqa_diamond_0035",
    "gpqa_gpqa_diamond_0037",
    "gpqa_gpqa_diamond_0039",
]

# Module-level globals — initialized in __main__ guard to avoid
# re-initialization in vLLM's spawned child processes.
llm = None
tokenizer = None
END_THINK_TOKENS = None


class LastNBaselineMethod(BaseMethod):
    """Trivial baseline: select the last N sentences of the compression region."""

    def __init__(self):
        super().__init__("last_n_baseline")

    def infer(self, data, verbose=True):
        """No LLM call — just pick the last target_num_sentences of the region."""
        rows = data if isinstance(data, list) else [data]
        results = []
        for row in rows:
            region_start = row.get("region_start_idx", 0)
            region_end = row.get("region_end_idx", 0)
            target_n = row.get("target_num_sentences", 5)
            char_budget = row.get("char_budget", 1000)

            num_region = region_end - region_start
            n_pick = min(target_n, num_region)
            selected = list(range(region_end - n_pick, region_end))

            sentences = row.get("sentences", [])
            while len(selected) > 1:
                total_chars = sum(len(sentences[i]) for i in selected)
                if total_chars <= char_budget:
                    break
                selected.pop(0)

            results.append({
                **row,
                "monitor_prompt": "",
                "monitor_response": f"last-{n_pick} baseline",
                "monitor_prediction": selected,
            })
        if verbose:
            print(f"Last-N baseline: selected {len(results)} rows")
        return results


# ── Helpers ──────────────────────────────────────────────────────────


def build_result_index(method_dir):
    """Scan method directory and return {question_id: count} of existing results."""
    index = {}
    if not os.path.isdir(method_dir):
        return index
    for name in os.listdir(method_dir):
        eval_path = os.path.join(method_dir, name, "compression_eval.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    ev = json.load(f)
                qid = ev.get("compressed_distribution", {}).get("question_id", "")
                if qid:
                    index[qid] = index.get(qid, 0) + 1
            except (json.JSONDecodeError, KeyError):
                pass
    return index


def get_existing_spec(task, qid, rollout_idx):
    """Load existing compression spec from disk, or None."""
    spec_dir = task.compression_dir / qid / f"rollout_{rollout_idx:03d}"
    if not spec_dir.exists():
        return None
    specs = sorted(spec_dir.rglob("compression_spec.json"), reverse=True)
    if not specs:
        return None
    with open(specs[0]) as f:
        d = json.load(f)
    return CompressionSpec(
        question_id=d["question_id"],
        rollout_idx=d.get("rollout_idx", rollout_idx),
        sentences=d["sentences"],
        region_start=d["region_start"],
        region_end=d["region_end"],
        region_type=d.get("region_type", "prefix"),
        target_num_sentences=d["target_num_sentences"],
        char_budget=d["char_budget"],
        compression_factor=d["compression_factor"],
        original_token_count=d.get("original_token_count", 0),
        region_token_count=d.get("region_token_count", 0),
    )


def build_monitor_row(spec, question):
    """Build a monitor data row from a CompressionSpec and Question."""
    row = {
        "question_id": spec.question_id,
        "question_type": getattr(question, "question_type", "multiple_choice"),
        "question": question.question,
        "full_cot": spec.full_cot,
        "sentences": spec.sentences,
        "region_sentences": spec.region_sentences,
        "region_start_idx": spec.region_start,
        "region_end_idx": spec.region_end,
        "region_type": spec.region_type,
        "target_num_sentences": spec.target_num_sentences,
        "char_budget": spec.char_budget,
        "compression_factor": spec.compression_factor,
        "rollout_idx": spec.rollout_idx,
        "original_token_count": spec.original_token_count,
        "region_token_count": spec.region_token_count,
    }
    if isinstance(question, GPQAQuestion):
        row["choices"] = question.choices
        row["correct_answer"] = question.correct_answer
    elif hasattr(question, "bad_outcome"):
        row["bad_outcome"] = question.bad_outcome
    return row


# ── vLLM batch evaluation ───────────────────────────────────────────


def batch_evaluate_compressions(task, question_id, jobs, num_resamples=50,
                                 temperature=0.7):
    """
    Batch-evaluate ALL compressed CoTs for one question via vLLM.

    Uses vLLM native batching instead of ThreadPoolExecutor:
      - Step 1: thinking continuation (prefix jobs, n=num_resamples per prompt)
      - Step 2: answer generation (one per continuation, n=1)
      - Middle mode: single-step with n=num_resamples

    Returns list of distribution dicts, one per job.
    """
    if not jobs:
        return []

    loaded = task.load_question_and_cot(question_id, jobs[0]["rollout_idx"])
    if loaded is None:
        raise RuntimeError(f"Could not load question for {question_id}")
    question, _ = loaded

    user_msg = task._user_msg(question)
    ANSWER_MAX_TOKENS = 256

    # Pre-compute tokenized prompts for each job
    prepared = []
    for job in jobs:
        prompt_str = build_thinking_prompt(
            tokenizer, user_msg, cot_prefix=job["compressed_cot"],
        )
        tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        spec = job["spec"]
        cb = spec.original_token_count - spec.region_token_count
        if cb <= 0:
            cb = None

        prepared.append({"tokens": tokens, "continuation_budget": cb})

    job_answers = [[] for _ in range(len(jobs))]

    # Separate prefix-mode and middle-mode jobs by index
    prefix_indices = [i for i, p in enumerate(prepared) if p["continuation_budget"] is not None]
    middle_indices = [i for i, p in enumerate(prepared) if p["continuation_budget"] is None]

    # ── Prefix mode: chunked two-step generation ──
    # Process jobs in chunks so each chunk does Step 1 → Step 2 while
    # KV cache is warm, but with enough sequences per call to saturate GPUs.
    CHUNK_SIZE = 15  # 15 jobs × 50 resamples = 750 sequences per call
    if prefix_indices:
        n_chunks = (len(prefix_indices) + CHUNK_SIZE - 1) // CHUNK_SIZE
        tqdm.write(f"    Prefix mode: {len(prefix_indices)} jobs "
                    f"x {num_resamples} resamples ({n_chunks} chunks of <={CHUNK_SIZE})")

        step2_params = VllmSamplingParams(
            max_tokens=ANSWER_MAX_TOKENS, temperature=temperature, n=1,
        )

        for chunk_start in tqdm(
            range(0, len(prefix_indices), CHUNK_SIZE),
            desc="    Eval chunks", leave=False,
            total=n_chunks,
        ):
            chunk = prefix_indices[chunk_start:chunk_start + CHUNK_SIZE]

            # Step 1: thinking continuations for all jobs in this chunk
            step1_prompts = [
                {"prompt_token_ids": prepared[pj_idx]["tokens"]}
                for pj_idx in chunk
            ]
            step1_params = [
                VllmSamplingParams(
                    max_tokens=prepared[pj_idx]["continuation_budget"],
                    temperature=temperature, n=num_resamples,
                )
                for pj_idx in chunk
            ]
            step1_outputs = llm.generate(step1_prompts, step1_params,
                                          use_tqdm=False)

            # Step 2: answer generation for all continuations in this chunk
            step2_inputs = []
            step2_job_indices = []
            for ci, pj_idx in enumerate(chunk):
                base_tokens = prepared[pj_idx]["tokens"]
                for completion in step1_outputs[ci].outputs:
                    think_token_ids = list(completion.token_ids)
                    full_prefix = base_tokens + think_token_ids + END_THINK_TOKENS
                    step2_inputs.append({"prompt_token_ids": full_prefix})
                    step2_job_indices.append(pj_idx)

            step2_outputs = llm.generate(step2_inputs, step2_params,
                                          use_tqdm=False)

            for s2_idx, output in enumerate(step2_outputs):
                orig_idx = step2_job_indices[s2_idx]
                answer_text = output.outputs[0].text.strip()
                answer = task._extract_answer_from_text(answer_text, question)
                if answer:
                    job_answers[orig_idx].append(answer)

    # ── Middle mode: single-step generation ──
    if middle_indices:
        mid_prompts = [{"prompt_token_ids": prepared[i]["tokens"]} for i in middle_indices]
        mid_params = VllmSamplingParams(
            max_tokens=2048, temperature=temperature, n=num_resamples,
        )

        tqdm.write(f"    Middle mode: {len(middle_indices)} prompts "
                    f"x {num_resamples} resamples")
        mid_outputs = llm.generate(mid_prompts, mid_params, use_tqdm=False)

        for mi, mj_idx in enumerate(middle_indices):
            output = mid_outputs[mi]
            for completion in output.outputs:
                answer, _, _ = task._extract_answer(
                    list(completion.token_ids), tokenizer, question,
                )
                if answer:
                    job_answers[mj_idx].append(answer)

    # Compute distributions
    distributions = []
    for job_idx, answers in enumerate(job_answers):
        counts = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        total = len(answers)
        dist = {k: v / total for k, v in counts.items()} if total > 0 else {}
        most_common = max(counts.items(), key=lambda x: x[1]) if counts else ("", 0)
        distributions.append({
            "question_id": question_id,
            "num_resamples": num_resamples,
            "valid_answers": total,
            "answer_counts": counts,
            "distribution": dist,
            "most_common": most_common[0],
            "agreement_rate": most_common[1] / total if total > 0 else 0,
            "region_type": jobs[job_idx]["spec"].region_type,
            "continuation_budget": prepared[job_idx]["continuation_budget"],
        })

    return distributions


def compute_baseline_vllm(task, question_id, rollout_idx=0, num_resamples=50,
                           temperature=0.7):
    """
    Compute baseline answer distribution using vLLM with full (uncompressed) CoT.

    Same two-step logic as batch_evaluate_compressions but for a single
    question with the original CoT as prefix. Returns the same dict format
    as task.get_baseline_distribution().
    """
    loaded = task.load_question_and_cot(question_id, rollout_idx)
    if loaded is None:
        raise RuntimeError(f"Could not load question for {question_id}")
    question, source_cot = loaded

    # Load spec for region_type and token counts
    spec_data = task._load_latest_spec(question_id, rollout_idx)
    region_type = spec_data.get("region_type", "middle") if spec_data else "middle"

    continuation_budget = None
    if region_type == "prefix" and spec_data:
        original_token_count = spec_data.get("original_token_count", 0)
        region_token_count = spec_data.get("region_token_count", 0)
        if original_token_count > 0 and region_token_count > 0:
            continuation_budget = original_token_count - region_token_count

    user_msg = task._user_msg(question)
    prompt_str = build_thinking_prompt(tokenizer, user_msg, cot_prefix=source_cot)
    tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

    ANSWER_MAX_TOKENS = 256
    answers = []

    if continuation_budget is not None:
        # Prefix mode: two-step generation
        step1_params = VllmSamplingParams(
            max_tokens=continuation_budget, temperature=temperature,
            n=num_resamples,
        )
        step1_output = llm.generate(
            [{"prompt_token_ids": tokens}], step1_params,
            use_tqdm=False,
        )[0]

        step2_inputs = []
        for completion in step1_output.outputs:
            think_token_ids = list(completion.token_ids)
            full_prefix = tokens + think_token_ids + END_THINK_TOKENS
            step2_inputs.append({"prompt_token_ids": full_prefix})

        step2_params = VllmSamplingParams(
            max_tokens=ANSWER_MAX_TOKENS, temperature=temperature, n=1,
        )
        step2_outputs = llm.generate(step2_inputs, step2_params,
                                      use_tqdm=False)

        for output in step2_outputs:
            answer_text = output.outputs[0].text.strip()
            answer = task._extract_answer_from_text(answer_text, question)
            if answer:
                answers.append(answer)
    else:
        # Middle mode: single-step generation
        params = VllmSamplingParams(
            max_tokens=2048, temperature=temperature, n=num_resamples,
        )
        output = llm.generate([{"prompt_token_ids": tokens}], params,
                               use_tqdm=False)[0]

        for completion in output.outputs:
            answer, _, _ = task._extract_answer(
                list(completion.token_ids), tokenizer, question,
            )
            if answer:
                answers.append(answer)

    # Build distribution dict (same format as task.get_baseline_distribution)
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


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── vLLM initialization (multi-GPU auto-detect) ──────────────────
    NUM_GPUS = torch.cuda.device_count() or 1

    llm = LLM(
        model=SUBJECT_MODEL,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=NUM_GPUS,
        max_model_len=16384,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_num_seqs=512,
    )
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)
    END_THINK_TOKENS = tokenizer.encode("</think>", add_special_tokens=False)

    task = CompressedCotTask(
        model=SUBJECT_MODEL,
        compression_factor=COMPRESSION_FACTOR,
        char_limit_multiplier=CHAR_LIMIT_MULTIPLIER,
        compress_pct=COMPRESS_PCT,
        region=REGION,
    )

    run_bb = "--skip-bb" not in sys.argv
    run_faithful = "--skip-faithful" not in sys.argv
    run_baseline = "--skip-baseline" not in sys.argv

    # Build skip index from existing results
    BB_DIR = str(task.data_dir / "llm_monitor_sentence_selection")
    FB_DIR = str(task.data_dir / "llm_monitor_faithful_sentence_selection")
    LN_DIR = str(task.data_dir / "last_n_baseline")

    tqdm.write("Scanning existing results...")
    bb_index = build_result_index(BB_DIR) if run_bb else {}
    fb_index = build_result_index(FB_DIR) if run_faithful else {}
    ln_index = build_result_index(LN_DIR) if run_baseline else {}

    for qid in QUESTION_IDS:
        bb_n = bb_index.get(qid, 0)
        fb_n = fb_index.get(qid, 0)
        ln_n = ln_index.get(qid, 0)
        tqdm.write(f"  {qid}: BB={bb_n}, Faithful={fb_n}, Last-N={ln_n}")

    for q_idx, qid in enumerate(QUESTION_IDS):
        # Determine which methods need to run
        need_bb = run_bb and bb_index.get(qid, 0) < NUM_ROLLOUTS
        need_fb = run_faithful and fb_index.get(qid, 0) < NUM_ROLLOUTS
        need_ln = run_baseline and ln_index.get(qid, 0) < NUM_ROLLOUTS

        if not need_bb and not need_fb and not need_ln:
            tqdm.write(f"\n[{q_idx+1}/{len(QUESTION_IDS)}] {qid} — skipping (already complete)")
            continue

        methods_str = ", ".join(
            [m for m, needed in [("BB", need_bb), ("Faithful", need_fb), ("Last-N", need_ln)] if needed]
        )
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"[{q_idx+1}/{len(QUESTION_IDS)}] {qid}  [{methods_str}]")
        tqdm.write(f"{'='*60}")

        # Load or compute baseline distribution
        baseline_path = task.compression_dir / qid / "baseline_distribution.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
        else:
            tqdm.write("  Computing baseline distribution (vLLM)...")
            baseline = compute_baseline_vllm(
                task, qid, rollout_idx=0, num_resamples=NUM_RESAMPLES,
            )
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, "w") as f:
                json.dump(baseline, f, indent=2)
            tqdm.write(f"  Saved baseline to {baseline_path}")

        # ── Phase 1: Prepare all specs + monitor predictions (parallel) ──
        def prepare_one_rollout(rollout_idx):
            """Prepare spec + run monitors for one rollout. Returns list of jobs."""
            spec = get_existing_spec(task, qid, rollout_idx)
            if spec is None:
                spec = task.run_data(question_id=qid, rollout_idx=rollout_idx, verbose=False)
            if spec is None:
                return []

            loaded = task.load_question_and_cot(qid, rollout_idx)
            if loaded is None:
                return []
            question, _ = loaded
            monitor_row = build_monitor_row(spec, question)
            monitor_data = [monitor_row]

            rollout_jobs = []

            if need_bb:
                prompt = SentenceSelectionPrompt()
                monitor = LlmMonitor(prompt=prompt, model=MONITOR_MODEL)
                monitor.set_task(task)
                results = monitor.infer(monitor_data, verbose=False)
                selected = results[0].get("monitor_prediction") or []
                relative = [i - spec.region_start for i in selected]
                compressed_cot = spec.reconstruct_from_indices(relative)
                rollout_jobs.append({
                    "rollout_idx": rollout_idx, "method": "bb",
                    "compressed_cot": compressed_cot,
                    "selected_indices": selected, "relative_indices": relative,
                    "output_folder": monitor.get_folder(), "method_obj": monitor,
                    "spec": spec,
                })

            if need_fb:
                prompt = FaithfulSentenceSelectionPrompt()
                monitor = LlmMonitor(prompt=prompt, model=MONITOR_MODEL)
                monitor.set_task(task)
                results = monitor.infer(monitor_data, verbose=False)
                selected = results[0].get("monitor_prediction") or []
                relative = [i - spec.region_start for i in selected]
                compressed_cot = spec.reconstruct_from_indices(relative)
                rollout_jobs.append({
                    "rollout_idx": rollout_idx, "method": "faithful",
                    "compressed_cot": compressed_cot,
                    "selected_indices": selected, "relative_indices": relative,
                    "output_folder": monitor.get_folder(), "method_obj": monitor,
                    "spec": spec,
                })

            if need_ln:
                last_n = LastNBaselineMethod()
                last_n.set_task(task)
                results = last_n.infer(monitor_data, verbose=False)
                selected = results[0].get("monitor_prediction") or []
                relative = [i - spec.region_start for i in selected]
                compressed_cot = spec.reconstruct_from_indices(relative)
                rollout_jobs.append({
                    "rollout_idx": rollout_idx, "method": "last_n",
                    "compressed_cot": compressed_cot,
                    "selected_indices": selected, "relative_indices": relative,
                    "output_folder": last_n.get_folder(), "method_obj": last_n,
                    "spec": spec,
                })

            return rollout_jobs

        jobs = []
        with ThreadPoolExecutor(max_workers=NUM_ROLLOUTS) as executor:
            futures = {
                executor.submit(prepare_one_rollout, ri): ri
                for ri in range(NUM_ROLLOUTS)
            }
            for future in tqdm(
                as_completed(futures), total=NUM_ROLLOUTS,
                desc="  Preparing", unit="rollout", leave=False,
            ):
                rollout_jobs = future.result()
                jobs.extend(rollout_jobs)
        tqdm.write(f"  Prepared {len(jobs)} compression jobs")

        # ── Phase 2: Batch evaluate ALL compressions via vLLM ──
        distributions = batch_evaluate_compressions(
            task, qid, jobs,
            num_resamples=NUM_RESAMPLES,
        )

        # ── Phase 3: Compute KL, save results ──
        bb_eval_results = []
        faithful_eval_results = []
        last_n_eval_results = []

        for job, dist in zip(jobs, distributions):
            metrics = task.evaluate(
                [dist["distribution"]], [baseline["distribution"]],
                mode=EVAL_MODE,
            )

            eval_result = {
                "mode": EVAL_MODE,
                "selected_indices": job["selected_indices"],
                "relative_indices": job["relative_indices"],
                "compressed_distribution": dist,
                "baseline_distribution": baseline,
                **metrics,
            }

            # Save per-rollout results
            output_folder = Path(job["output_folder"])
            with open(output_folder / "compression_eval.json", "w") as f:
                json.dump(eval_result, f, indent=2)
            with open(output_folder / "compressed_cot.txt", "w") as f:
                f.write(job["compressed_cot"])

            if hasattr(job["method_obj"], "_output") and job["method_obj"]._output:
                job["method_obj"]._output.mark_success()

            record = {"rollout_idx": job["rollout_idx"], **eval_result}
            if job["method"] == "bb":
                bb_eval_results.append(record)
            elif job["method"] == "faithful":
                faithful_eval_results.append(record)
            elif job["method"] == "last_n":
                last_n_eval_results.append(record)

        # Print question summary
        for method_name, results in [("BB Monitor", bb_eval_results),
                                      ("Faithful BB", faithful_eval_results),
                                      ("Last-N", last_n_eval_results)]:
            if results:
                avg_kl = sum(r["kl_divergence"] for r in results) / len(results)
                tqdm.write(f"  {method_name}: avg KL={avg_kl:.4f} ({len(results)} rollouts)")

        # Save aggregate results per question
        if bb_eval_results:
            summary_path = task.compression_dir / qid / "all_rollouts_bb_eval.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(bb_eval_results, f, indent=2)
        if faithful_eval_results:
            summary_path = task.compression_dir / qid / "all_rollouts_faithful_eval.json"
            with open(summary_path, "w") as f:
                json.dump(faithful_eval_results, f, indent=2)
        if last_n_eval_results:
            summary_path = task.compression_dir / qid / "all_rollouts_last_n_eval.json"
            with open(summary_path, "w") as f:
                json.dump(last_n_eval_results, f, indent=2)

    tqdm.write("\nDone.")
