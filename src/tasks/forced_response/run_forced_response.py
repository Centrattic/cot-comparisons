#!/usr/bin/env python3
"""
Main script for running the Forced Response task.

Usage:
    # Run verification to find a high-agreement question
    python run_forced_response.py verify --num-rollouts 50

    # Run true forcing (Tinker prefill) on a verified question
    python run_forced_response.py force --question-id gpqa_sample_001 -n 5

    # Run forcing monitor on a verified question
    python run_forced_response.py monitor-forcing --question-id gpqa_sample_001 -n 5

    # Run resampling monitor on a verified question
    python run_forced_response.py monitor-resampling --question-id gpqa_sample_001 -n 5

    # Full pipeline: verify then force
    python run_forced_response.py full --num-rollouts 50 --num-forces 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from src.tasks.forced_response.data_loader import (
    GPQAQuestion,
    load_custom_questions,
    load_gpqa_from_huggingface,
    load_gpqa_questions,
)
from src.tasks.forced_response.forcing import (
    run_forcing,
    run_forcing_from_verification,
)
from src.tasks.forced_response.monitors import (
    run_monitor_forcing,
    run_monitor_forcing_from_verification,
    run_monitor_resampling,
    run_monitor_resampling_from_verification,
)
from src.tasks.forced_response.resampling import (
    run_resampling_from_verification,
)
from src.tasks.forced_response.task import ForcedResponseTask
from src.tasks.forced_response.verification import (
    find_high_agreement_question,
    find_low_agreement_question,
    run_verification,
)

DEFAULT_MODEL = "moonshotai/Kimi-K2-Thinking"


def cmd_verify(args):
    """Run verification to find high-agreement questions."""
    print(f"Running verification with {args.num_rollouts} rollouts")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    if getattr(args, "max_agreement", None) is not None:
        print(f"Looking for LOW agreement (max: {args.max_agreement})")
    else:
        print(f"Agreement threshold: {args.threshold}")
    print()

    # Load questions
    if args.use_custom:
        print("Loading custom questions...")
        questions = load_custom_questions()
        if not questions:
            print("No custom questions found in questions.json")
            return 1
    elif args.use_huggingface:
        print("Loading questions from HuggingFace...")
        questions = load_gpqa_from_huggingface(
            subset=args.gpqa_subset,
            max_questions=args.max_questions,
        )
    else:
        print("Using sample GPQA questions...")
        questions = load_gpqa_questions(use_samples=True)

    if args.question_id:
        # Verify a specific question
        question = next((q for q in questions if q.id == args.question_id), None)
        if not question:
            print(f"Question {args.question_id} not found")
            return 1

        summary = run_verification(
            question=question,
            num_rollouts=args.num_rollouts,
            model=args.model,
            api_key=args.api_key,
            max_workers=args.max_workers,
            verbose=True,
        )
        return 0 if summary["meets_threshold"] else 1

    elif getattr(args, "max_agreement", None) is not None:
        # Find a low-agreement question
        question = find_low_agreement_question(
            questions=questions,
            num_rollouts=args.num_rollouts,
            max_agreement=args.max_agreement,
            model=args.model,
            api_key=args.api_key,
            max_workers=args.max_workers,
            verbose=True,
        )

        if question:
            print(f"\nLow-agreement question found: {question.id}")
            return 0
        else:
            print("\nNo low-agreement question found")
            return 1

    else:
        # Find first high-agreement question
        question = find_high_agreement_question(
            questions=questions,
            num_rollouts=args.num_rollouts,
            threshold=args.threshold,
            model=args.model,
            api_key=args.api_key,
            max_workers=args.max_workers,
            verbose=True,
        )

        if question:
            print(f"\nHigh-agreement question found: {question.id}")
            return 0
        else:
            print("\nNo high-agreement question found")
            return 1


def cmd_force(args):
    """Run true forcing on a verified question."""
    if not args.question_id:
        # Find verified questions
        task = ForcedResponseTask(model=args.model)
        verified = task.get_verified_questions(threshold=args.threshold)

        if not verified:
            print("No verified questions found. Run 'verify' first.")
            return 1

        args.question_id = verified[0]
        print(f"Using verified question: {args.question_id}")

    if args.openrouter:
        # Use OpenRouter with DeepInfra fp4
        import asyncio

        summary = asyncio.run(
            run_forcing_openrouter(
                question_id=args.question_id,
                rollout_idx=args.rollout_idx,
                num_forces=args.num_forces,
                max_sentences=args.max_sentences,
                max_workers=args.max_workers,
                verbose=True,
            )
        )
    else:
        # Use Tinker
        print(
            f"Running true forcing (Tinker) with {args.num_forces} samples per sentence"
        )
        if args.max_sentences:
            print(f"Limiting to first {args.max_sentences} sentences")
        print(f"Model: {args.model}")
        print()

        summary = run_forcing_from_verification(
            question_id=args.question_id,
            rollout_idx=args.rollout_idx,
            num_forces=args.num_forces,
            max_sentences=args.max_sentences,
            model=args.model,
            verbose=True,
        )

    if summary:
        print("\n" + "=" * 60)
        method = "OpenRouter" if args.openrouter else "Tinker"
        print(f"FORCING COMPLETE ({method})")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        if "correct_answer" in summary:
            print(f"Correct answer: {summary['correct_answer']}")
        print(f"Sentences processed: {summary['num_sentences']}")

        # Show answer progression
        print("\nAnswer progression by sentence:")
        for result in summary["sentence_results"]:
            idx = result["sentence_idx"]
            counts = result.get("answer_counts", {})
            most_common = result.get("most_common", "?")
            print(f"  Sentence {idx + 1}: {counts} (most common: {most_common})")

        return 0
    else:
        return 1


def cmd_monitor_forcing(args):
    """Run forcing monitor on a verified question."""
    if not args.question_id:
        task = ForcedResponseTask(model=args.model)
        verified = task.get_verified_questions(threshold=args.threshold)

        if not verified:
            print("No verified questions found. Run 'verify' first.")
            return 1

        args.question_id = verified[0]
        print(f"Using verified question: {args.question_id}")

    print(f"Running monitor-forcing with {args.num_forces} attempts per sentence")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print()

    summary = run_monitor_forcing_from_verification(
        question_id=args.question_id,
        rollout_idx=args.rollout_idx,
        num_forces=args.num_forces,
        max_workers=args.max_workers,
        model=args.model,
        api_key=args.api_key,
        verbose=True,
    )

    if summary:
        print("\n" + "=" * 60)
        print("MONITOR-FORCING COMPLETE")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        if "correct_answer" in summary:
            print(f"Correct answer: {summary['correct_answer']}")
        else:
            print(f"Bad outcome: {summary.get('bad_outcome', '?')}")
        print(f"Sentences processed: {summary['num_sentences']}")

        print("\nPredicted distributions by sentence:")
        for result in summary["sentence_results"]:
            idx = result["sentence_idx"]
            dist = result.get("distribution", {})
            most_likely = result.get("most_likely", "?")
            is_valid = result.get("is_valid", False)
            # Format distribution nicely
            dist_str = " ".join(f"{k}:{v:.0%}" for k, v in sorted(dist.items()) if v > 0)
            status = "✓" if is_valid else "✗"
            print(f"  Sentence {idx + 1}: [{dist_str}] → {most_likely} {status}")

        return 0
    else:
        return 1


def cmd_monitor_resampling(args):
    """Run resampling monitor on a verified question."""
    if not args.question_id:
        task = ForcedResponseTask(model=args.model)
        verified = task.get_verified_questions(threshold=args.threshold)

        if not verified:
            print("No verified questions found. Run 'verify' first.")
            return 1

        args.question_id = verified[0]
        print(f"Using verified question: {args.question_id}")

    print(
        f"Running monitor-resampling at ~{args.num_prefix_points} prefix points (predicting distribution over {args.num_resamples} resamples)"
    )
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print()

    summary = run_monitor_resampling_from_verification(
        question_id=args.question_id,
        rollout_idx=args.rollout_idx,
        num_resamples=args.num_resamples,
        num_prefix_points=args.num_prefix_points,
        max_workers=args.max_workers,
        model=args.model,
        api_key=args.api_key,
        verbose=True,
    )

    if summary:
        print("\n" + "=" * 60)
        print("MONITOR-RESAMPLING COMPLETE")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        if "correct_answer" in summary:
            print(f"Correct answer: {summary['correct_answer']}")
        else:
            print(f"Bad outcome: {summary.get('bad_outcome', '?')}")
        print(
            f"Prefix points: {summary['num_prefix_points']} (stride {summary['stride']}, {summary['num_sentences']} total sentences)"
        )

        print("\nPredicted distributions by prefix point:")
        for result in summary["sentence_results"]:
            idx = result["sentence_idx"]
            dist = result.get("distribution", {})
            most_likely = result.get("most_likely", "?")
            is_valid = result.get("is_valid", False)
            # Format distribution nicely
            dist_str = " ".join(f"{k}:{v:.0%}" for k, v in sorted(dist.items()) if v > 0)
            status = "✓" if is_valid else "✗"
            print(f"  Sentence {idx + 1}: [{dist_str}] → {most_likely} {status}")

        return 0
    else:
        return 1


def cmd_full(args):
    """Run full pipeline: verify then force."""
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)

    # Step 1: Verify
    print("\nStep 1: Finding high-agreement question...")
    print()

    if getattr(args, "use_custom", False):
        questions = load_custom_questions()
        if not questions:
            print("No custom questions found in questions.json")
            return 1
    elif args.use_huggingface:
        questions = load_gpqa_from_huggingface(
            subset=args.gpqa_subset,
            max_questions=args.max_questions,
        )
    else:
        questions = load_gpqa_questions(use_samples=True)

    question = find_high_agreement_question(
        questions=questions,
        num_rollouts=args.num_rollouts,
        threshold=args.threshold,
        model=args.model,
        api_key=args.api_key,
        max_workers=getattr(args, "max_workers", 250),
        verbose=True,
    )

    if not question:
        print("\nNo high-agreement question found. Stopping.")
        return 1

    # Step 2: Force
    print("\n" + "=" * 60)
    print("Step 2: Running forcing...")
    print("=" * 60 + "\n")

    summary = run_forcing_from_verification(
        question_id=question.id,
        rollout_idx=args.rollout_idx,
        num_forces=args.num_forces,
        model=args.model,
        verbose=True,
    )

    if summary:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        if "correct_answer" in summary:
            print(f"Correct answer: {summary['correct_answer']}")
        else:
            print(f"Bad outcome: {summary.get('bad_outcome', '?')}")
        print(f"Sentences processed: {summary['num_sentences']}")
        return 0
    else:
        return 1


def cmd_resample(args):
    """Run resampling (Tinker prefix continuation) on a verified question."""
    if not args.question_id:
        task = ForcedResponseTask(model=args.model)
        verified = task.get_verified_questions(threshold=args.threshold)

        if not verified:
            print("No verified questions found. Run 'verify' first.")
            return 1

        args.question_id = verified[0]
        print(f"Using verified question: {args.question_id}")

    print(
        f"Running resampling (Tinker) with {args.num_resamples} samples at ~{args.num_prefix_points} prefix points"
    )
    print(f"Model: {args.model}")
    print()

    summary = run_resampling_from_verification(
        question_id=args.question_id,
        model=args.model,
        rollout_idx=args.rollout_idx,
        num_resamples=args.num_resamples,
        num_prefix_points=args.num_prefix_points,
        verbose=True,
    )

    if summary:
        print("\n" + "=" * 60)
        print("RESAMPLING COMPLETE (Tinker)")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        if "correct_answer" in summary:
            print(f"Correct answer: {summary['correct_answer']}")
        else:
            print(f"Bad outcome: {summary.get('bad_outcome', '?')}")
        print(
            f"Prefix points: {summary['num_prefix_points']} (stride {summary['stride']}, {summary['num_sentences']} total sentences)"
        )
        print(f"Resamples per point: {summary['num_resamples']}")

        print("\nAnswer distribution by prefix point:")
        for result in summary["sentence_results"]:
            idx = result["sentence_idx"]
            counts = result["answer_counts"]
            valid = result["valid_answers"]
            total = result["total_resamples"]
            most_common = result.get("most_common", "?")
            rate = result.get("agreement_rate", 0)
            print(
                f"  Sentence {idx + 1}: {counts} (valid: {valid}/{total}, most common: {most_common} @ {rate:.0%})"
            )

        return 0
    else:
        return 1


def cmd_list(args):
    """List verified questions and their status."""
    task = ForcedResponseTask(model=args.model)

    print("Verified Questions:")
    print("-" * 60)

    if not task.verification_dir.exists():
        print("No verification data found.")
        return 0

    for question_dir in sorted(task.verification_dir.iterdir()):
        if question_dir.is_dir():
            summary = task.load_verification_summary(question_dir.name)
            if summary:
                agreement = summary.get("agreement_rate", 0)
                meets = "YES" if summary.get("meets_threshold", False) else "NO"
                most_common = summary.get("most_common_answer", "?")
                question_type = summary.get("question_type", "multiple_choice")
                print(f"  {question_dir.name}:")
                print(f"    Agreement: {agreement:.1%} (meets threshold: {meets})")
                if question_type == "binary_judge":
                    bad_rate = summary.get("bad_outcome_rate", 0)
                    bad_outcome = summary.get("bad_outcome", "?")
                    print(f"    Most common: {most_common} (bad outcome '{bad_outcome}' rate: {bad_rate:.1%})")
                else:
                    correct = "correct" if summary.get("is_correct", False) else "incorrect"
                    print(f"    Most common: {most_common} ({correct})")

    # Show results for each mode
    modes = [
        ("Forcing (Tinker)", task.forcing_dir),
        ("Monitor-Forcing", task.monitor_forcing_dir),
        ("Resampling (Tinker)", task.resampling_dir),
        ("Monitor-Resampling", task.monitor_resampling_dir),
    ]
    for label, data_dir in modes:
        print()
        print(f"{label} Results:")
        print("-" * 60)

        if not data_dir.exists():
            print(f"  No data found.")
            continue

        for question_dir in sorted(data_dir.iterdir()):
            if question_dir.is_dir():
                rollout_dirs = sorted(
                    [
                        d
                        for d in question_dir.iterdir()
                        if d.is_dir() and d.name.startswith("rollout_")
                    ]
                )
                if rollout_dirs:
                    print(f"  {question_dir.name}:")
                    for rollout_dir in rollout_dirs:
                        # Find latest timestamped run
                        latest = task.get_latest_run_dir(rollout_dir)
                        if latest and (latest / "summary.json").exists():
                            with open(latest / "summary.json") as f:
                                summary = json.load(f)
                            num_sentences = summary.get("num_sentences", 0)
                            run_label = (
                                latest.name if latest != rollout_dir else "(legacy)"
                            )
                            print(
                                f"    {rollout_dir.name} [{run_label}]: {num_sentences} sentences"
                            )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the Forced Response task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run verification rollouts to find high-agreement questions",
    )
    verify_parser.add_argument(
        "--num-rollouts",
        "-n",
        type=int,
        default=50,
        help="Number of rollouts per question (default: 50)",
    )
    verify_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Agreement threshold (default: 0.8)",
    )
    verify_parser.add_argument(
        "--question-id",
        "-q",
        help="Verify a specific question ID",
    )
    verify_parser.add_argument(
        "--use-custom",
        action="store_true",
        help="Load custom questions from questions.json",
    )
    verify_parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Load questions from HuggingFace instead of samples",
    )
    verify_parser.add_argument(
        "--gpqa-subset",
        default="gpqa_diamond",
        help="GPQA subset to use (default: gpqa_diamond)",
    )
    verify_parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum questions to try",
    )
    verify_parser.add_argument(
        "--max-agreement",
        type=float,
        help="Find a LOW-agreement question (no answer exceeds this rate, e.g. 0.5)",
    )
    verify_parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=250,
        help="Maximum concurrent API calls (default: 250)",
    )
    verify_parser.set_defaults(func=cmd_verify)

    # Force command (Tinker-based true forcing)
    force_parser = subparsers.add_parser(
        "force",
        help="Run true forcing (Tinker prefill) on a verified question",
    )
    force_parser.add_argument(
        "--question-id",
        "-q",
        help="Question ID to force (default: first verified question)",
    )
    force_parser.add_argument(
        "--rollout-idx",
        "-r",
        type=int,
        default=0,
        help="Which verification rollout to use as source (default: 0)",
    )
    force_parser.add_argument(
        "--num-forces",
        "-n",
        type=int,
        default=5,
        help="Number of force samples per sentence (default: 5)",
    )
    force_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Agreement threshold for selecting questions (default: 0.8)",
    )
    force_parser.add_argument(
        "--max-sentences",
        "-m",
        type=int,
        default=None,
        help="Only force the first M sentences (default: all)",
    )
    force_parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter (DeepInfra fp4) instead of Tinker",
    )
    force_parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=300,
        help="Max concurrent requests for OpenRouter mode (default: 300)",
    )
    force_parser.set_defaults(func=cmd_force)

    # Monitor-forcing command
    mf_parser = subparsers.add_parser(
        "monitor-forcing",
        help="Run forcing monitor (predicts answer from prefill context)",
    )
    mf_parser.add_argument(
        "--question-id",
        "-q",
        help="Question ID to monitor (default: first verified question)",
    )
    mf_parser.add_argument(
        "--rollout-idx",
        "-r",
        type=int,
        default=0,
        help="Which verification rollout to use as source (default: 0)",
    )
    mf_parser.add_argument(
        "--num-forces",
        "-n",
        type=int,
        default=20,
        help="Number of forces to predict distribution over (default: 20)",
    )
    mf_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Agreement threshold for selecting questions (default: 0.8)",
    )
    mf_parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=300,
        help="Maximum concurrent API calls (default: 300)",
    )
    mf_parser.set_defaults(func=cmd_monitor_forcing)

    # Monitor-resampling command
    mr_parser = subparsers.add_parser(
        "monitor-resampling",
        help="Run resampling monitor (predicts answer distribution from prefix)",
    )
    mr_parser.add_argument(
        "--question-id",
        "-q",
        help="Question ID to monitor (default: first verified question)",
    )
    mr_parser.add_argument(
        "--rollout-idx",
        "-r",
        type=int,
        default=0,
        help="Which verification rollout to use as source (default: 0)",
    )
    mr_parser.add_argument(
        "--num-resamples",
        "-n",
        type=int,
        default=20,
        help="Number of hypothetical resamples to predict distribution over (default: 20)",
    )
    mr_parser.add_argument(
        "--num-prefix-points",
        type=int,
        default=20,
        help="Target number of evenly-spaced prefix points (default: 20)",
    )
    mr_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Agreement threshold for selecting questions (default: 0.8)",
    )
    mr_parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=300,
        help="Maximum concurrent API calls (default: 300)",
    )
    mr_parser.set_defaults(func=cmd_monitor_resampling)

    # Full command
    full_parser = subparsers.add_parser(
        "full",
        help="Run full pipeline: verify then force",
    )
    full_parser.add_argument(
        "--num-rollouts",
        type=int,
        default=50,
        help="Number of verification rollouts (default: 50)",
    )
    full_parser.add_argument(
        "--num-forces",
        type=int,
        default=5,
        help="Number of force attempts per sentence (default: 5)",
    )
    full_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Agreement threshold (default: 0.8)",
    )
    full_parser.add_argument(
        "--use-custom",
        action="store_true",
        help="Load custom questions from questions.json",
    )
    full_parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Load questions from HuggingFace instead of samples",
    )
    full_parser.add_argument(
        "--gpqa-subset",
        default="gpqa_diamond",
        help="GPQA subset to use (default: gpqa_diamond)",
    )
    full_parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum questions to try",
    )
    full_parser.add_argument(
        "--rollout-idx",
        "-r",
        type=int,
        default=0,
        help="Which verification rollout to use as source (default: 0)",
    )
    full_parser.set_defaults(func=cmd_full)

    # Resample command (Tinker-based prefix continuation)
    resample_parser = subparsers.add_parser(
        "resample",
        help="Run resampling (Tinker prefix continuation) on a verified question",
    )
    resample_parser.add_argument(
        "--question-id",
        "-q",
        help="Question ID to resample (default: first verified question)",
    )
    resample_parser.add_argument(
        "--rollout-idx",
        "-r",
        type=int,
        default=0,
        help="Which verification rollout to use as source (default: 0)",
    )
    resample_parser.add_argument(
        "--num-resamples",
        "-n",
        type=int,
        default=20,
        help="Number of continuations per prefix point (default: 20)",
    )
    resample_parser.add_argument(
        "--num-prefix-points",
        type=int,
        default=20,
        help="Target number of evenly-spaced prefix points (default: 20)",
    )
    resample_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Agreement threshold for selecting questions (default: 0.8)",
    )
    resample_parser.set_defaults(func=cmd_resample)

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List verified questions and forcing results",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    # Check for API key (not needed for Tinker-based commands or 'list')
    if args.command not in ("force", "resample", "list"):
        if not args.api_key and not os.environ.get("OPENROUTER_API_KEY"):
            print("Error: No API key provided. Set OPENROUTER_API_KEY or use --api-key")
            return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
