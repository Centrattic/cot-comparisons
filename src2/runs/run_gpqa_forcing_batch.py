#!/usr/bin/env python3
"""
Batch verification + forcing pipeline for GPQA Diamond questions.

Generates verification rollouts (source CoTs) then runs forcing across
all questions × rollouts in parallel via concurrent.futures.

Usage:
    python -m src2.runs.run_gpqa_forcing_batch
"""

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer  # eager import — avoid thread-unsafe lazy load

load_dotenv()

from src2.utils.questions import load_custom_questions, load_gpqa_from_huggingface
from src2.utils.verification import ensure_verification

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forced_response"
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"

SUBJECT_MODEL = "Qwen/Qwen3-32B"

# Questions
NUM_GPQA_QUESTIONS = 60  # load enough for 50 train + 10 eval
CUSTOM_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "utils" / "questions.json"
CUSTOM_IDS = ["custom_bagel_001", "starfish", "waffle"]  # also need 10 rollouts
EXCLUDE_IDS: List[str] = ["blackmail_001", "blackmail_mc_001"]

# Verification (source CoT generation)
NUM_VERIFICATION_ROLLOUTS = 10  # each becomes a forcing rollout
VERIFICATION_TEMPERATURE = 0.7
VERIFICATION_MAX_TOKENS = 8192
VERIFICATION_MAX_WORKERS = 300  # concurrent Tinker requests per question

# Forcing
MAX_SENTENCES = 30
SENTENCE_STRIDE = 1

# ──────────────────────────────────────────────────────────────────────


def run_single_forcing_job(
    forcing: "ForcingTask",
    question_id: str,
    rollout_idx: int,
    max_sentences: int,
    sentence_stride: int,
) -> dict:
    """Run forcing for one (question, rollout) pair."""
    result = forcing.run_data(
        question_id=question_id,
        rollout_idx=rollout_idx,
        max_sentences=max_sentences,
        sentence_stride=sentence_stride,
        verbose=False,
    )
    if result is None:
        return {
            "question_id": question_id,
            "rollout_idx": rollout_idx,
            "status": "skipped",
            "num_sentences": 0,
        }
    return {
        "question_id": question_id,
        "rollout_idx": rollout_idx,
        "status": "ok",
        "num_sentences": result.get("num_sentences", 0),
    }


def forcing_already_done(question_id: str, rollout_idx: int) -> bool:
    """Check if forcing data already exists for this (question, rollout)."""
    forcing_dir = DATA_DIR / "forcing" / question_id / f"rollout_{rollout_idx:03d}"
    if not forcing_dir.exists():
        return False
    summaries = list(forcing_dir.rglob("summary.json"))
    return len(summaries) > 0


def main():
    # ── Step 1: Load custom + GPQA Diamond questions ────────────────
    custom_questions = load_custom_questions(CUSTOM_QUESTIONS_FILE)
    custom_questions = [q for q in custom_questions if q.id in CUSTOM_IDS]
    print(f"Loaded {len(custom_questions)} custom questions: {[q.id for q in custom_questions]}")

    print(f"Loading {NUM_GPQA_QUESTIONS} GPQA Diamond questions from HuggingFace...")
    gpqa_questions = load_gpqa_from_huggingface(
        subset="gpqa_diamond", max_questions=NUM_GPQA_QUESTIONS,
    )
    gpqa_questions = [q for q in gpqa_questions if q.id not in EXCLUDE_IDS]
    print(f"  {len(gpqa_questions)} GPQA questions loaded")

    questions = custom_questions + gpqa_questions
    print(f"  {len(questions)} total questions")

    # ── Step 2: Run verification for all questions in parallel ──────
    print(f"\nStep 1/2: Verification ({NUM_VERIFICATION_ROLLOUTS} rollouts × {len(questions)} questions, all parallel)")
    print("=" * 60)

    def _verify_one(q):
        return q, ensure_verification(
            question=q,
            verification_dir=VERIFICATION_DIR,
            num_rollouts=NUM_VERIFICATION_ROLLOUTS,
            model=SUBJECT_MODEL,
            temperature=VERIFICATION_TEMPERATURE,
            max_tokens=VERIFICATION_MAX_TOKENS,
            max_workers=VERIFICATION_MAX_WORKERS,
        )

    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        futures = {executor.submit(_verify_one, q): q for q in questions}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verification"):
            q, summary = future.result()
            rate = summary["agreement_rate"]
            top = summary["most_common_answer"]
            valid = summary["valid_rollouts"]
            tqdm.write(f"  {q.id}: {valid}/{summary['total_rollouts']} valid, agreement={rate:.0%} ({top})")

    # ── Step 3: Build list of all (question, rollout) forcing jobs ─────
    print(f"\nStep 2/2: Forcing (rollouts 0-{NUM_VERIFICATION_ROLLOUTS - 1} per question)")
    print("=" * 60)

    jobs = []
    skipped = 0
    for q in questions:
        for rollout_idx in range(NUM_VERIFICATION_ROLLOUTS):
            if forcing_already_done(q.id, rollout_idx):
                skipped += 1
                continue
            jobs.append((q.id, rollout_idx))

    total_jobs = len(jobs) + skipped
    print(f"  {total_jobs} total (question, rollout) pairs")
    print(f"  {skipped} already done, {len(jobs)} to run")

    if not jobs:
        print("\nAll forcing jobs already complete!")
        _print_summary(questions)
        return

    # ── Step 4: Run all forcing jobs in parallel ──────────────────────
    from src2.tasks.forced_response.task import ForcingTask

    forcing = ForcingTask(model=SUBJECT_MODEL, data_dir=DATA_DIR)
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                run_single_forcing_job,
                forcing, qid, ridx,
                MAX_SENTENCES, SENTENCE_STRIDE,
            ): (qid, ridx)
            for qid, ridx in jobs
        }

        pbar = tqdm(total=len(futures), desc="Forcing")
        for future in as_completed(futures):
            qid, ridx = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result["status"]
                n_sent = result["num_sentences"]
                pbar.set_postfix_str(f"{qid} r{ridx}: {status} ({n_sent} sent)")
            except Exception as e:
                failed.append({"question_id": qid, "rollout_idx": ridx, "error": str(e)})
                tqdm.write(f"  FAILED: {qid} rollout {ridx}: {e}")
                traceback.print_exc()
            pbar.update(1)
        pbar.close()

    # ── Step 5: Summary ───────────────────────────────────────────────
    ok = [r for r in results if r["status"] == "ok"]
    skip = [r for r in results if r["status"] == "skipped"]
    print(f"\nDone: {len(ok)} succeeded, {len(skip)} skipped (no CoT), {len(failed)} failed")

    if failed:
        print("\nFailed jobs:")
        for f in failed[:10]:
            print(f"  {f['question_id']} rollout {f['rollout_idx']}: {f['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    _print_summary(questions)

    # Save run log
    log_dir = DATA_DIR / "answer_probe"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "num_questions": len(questions),
        "num_rollouts_per_question": NUM_VERIFICATION_ROLLOUTS,
        "total_jobs": total_jobs,
        "succeeded": len(ok),
        "skipped_existing": skipped,
        "skipped_no_cot": len(skip),
        "failed": len(failed),
        "failed_details": failed[:20],
    }
    with open(log_dir / "batch_forcing_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved to {log_dir / 'batch_forcing_log.json'}")


def _print_summary(questions):
    """Print per-question forcing coverage."""
    print(f"\nForcing coverage:")
    for q in questions:
        qdir = DATA_DIR / "forcing" / q.id
        if not qdir.exists():
            print(f"  {q.id}: no data")
            continue
        rollout_dirs = sorted(qdir.glob("rollout_*"))
        n_rollouts = 0
        total_sentences = 0
        for rd in rollout_dirs:
            sums = list(rd.rglob("summary.json"))
            if sums:
                n_rollouts += 1
                sentence_dirs = list(rd.rglob("sentence_*/summary.json"))
                total_sentences += len(sentence_dirs)
        print(f"  {q.id}: {n_rollouts} rollouts, {total_sentences} sentence dirs")


if __name__ == "__main__":
    main()
