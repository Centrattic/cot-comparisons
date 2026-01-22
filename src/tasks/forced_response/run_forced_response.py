#!/usr/bin/env python3
"""
Main script for running the Forced Response task.

Usage:
    # Run verification to find a high-agreement question
    python run_forced_response.py verify --num-rollouts 50

    # Run forcing on a verified question
    python run_forced_response.py force --question-id gpqa_sample_001

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
    load_gpqa_questions,
    load_gpqa_from_huggingface,
)
from src.tasks.forced_response.verification import (
    run_verification,
    find_high_agreement_question,
    find_low_agreement_question,
)
from src.tasks.forced_response.forcing import (
    run_forcing,
    run_forcing_from_verification,
)
from src.tasks.forced_response.resampling import (
    run_resampling,
    run_resampling_from_verification,
)
from src.tasks.forced_response.task import ForcedResponseTask


DEFAULT_MODEL = "moonshotai/kimi-k2-thinking"


def cmd_verify(args):
    """Run verification to find high-agreement questions."""
    print(f"Running verification with {args.num_rollouts} rollouts")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    if getattr(args, 'max_agreement', None) is not None:
        print(f"Looking for LOW agreement (max: {args.max_agreement})")
    else:
        print(f"Agreement threshold: {args.threshold}")
    print()

    # Load questions
    if args.use_huggingface:
        print("Loading questions from HuggingFace...")
        questions = load_gpqa_from_huggingface(
            subset=args.gpqa_subset,
            max_questions=args.max_questions,
        )
    else:
        print("Using sample questions...")
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

    elif getattr(args, 'max_agreement', None) is not None:
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
    """Run forcing on a verified question."""
    if not args.question_id:
        # Find verified questions
        task = ForcedResponseTask(model=args.model)
        verified = task.get_verified_questions(threshold=args.threshold)

        if not verified:
            print("No verified questions found. Run 'verify' first.")
            return 1

        args.question_id = verified[0]
        print(f"Using verified question: {args.question_id}")

    print(f"Running forcing with {args.num_forces} attempts per sentence")
    print(f"Model: {args.model}")
    print()

    summary = run_forcing_from_verification(
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
        print("FORCING COMPLETE")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        print(f"Correct answer: {summary['correct_answer']}")
        print(f"Sentences processed: {summary['num_sentences']}")

        # Show answer progression
        print("\nAnswer progression by sentence:")
        for result in summary["sentence_results"]:
            idx = result["sentence_idx"]
            counts = result["answer_counts"]
            valid = result["valid_single_token"]
            total = result["total_forces"]
            most_common = result.get("most_common", "?")
            print(f"  Sentence {idx + 1}: {counts} (valid: {valid}/{total}, most common: {most_common})")

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

    if args.use_huggingface:
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
        max_workers=getattr(args, 'max_workers', 250),
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
        rollout_idx=0,
        num_forces=args.num_forces,
        model=args.model,
        api_key=args.api_key,
        verbose=True,
    )

    if summary:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Question: {summary['question_id']}")
        print(f"Correct answer: {summary['correct_answer']}")
        print(f"Sentences processed: {summary['num_sentences']}")
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
                correct = "correct" if summary.get("is_correct", False) else "incorrect"
                most_common = summary.get("most_common_answer", "?")
                print(f"  {question_dir.name}:")
                print(f"    Agreement: {agreement:.1%} (meets threshold: {meets})")
                print(f"    Most common: {most_common} ({correct})")

    print()
    print("Forcing Results:")
    print("-" * 60)

    if not task.forcing_dir.exists():
        print("No forcing data found.")
        return 0

    for question_dir in sorted(task.forcing_dir.iterdir()):
        if question_dir.is_dir():
            summary_path = question_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                num_sentences = summary.get("num_sentences", 0)
                print(f"  {question_dir.name}: {num_sentences} sentences processed")

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
        "--num-rollouts", "-n",
        type=int,
        default=50,
        help="Number of rollouts per question (default: 50)",
    )
    verify_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.8,
        help="Agreement threshold (default: 0.8)",
    )
    verify_parser.add_argument(
        "--question-id", "-q",
        help="Verify a specific question ID",
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
        "--max-workers", "-w",
        type=int,
        default=250,
        help="Maximum concurrent API calls (default: 250)",
    )
    verify_parser.set_defaults(func=cmd_verify)

    # Force command
    force_parser = subparsers.add_parser(
        "force",
        help="Run forcing on a verified question",
    )
    force_parser.add_argument(
        "--question-id", "-q",
        help="Question ID to force (default: first verified question)",
    )
    force_parser.add_argument(
        "--rollout-idx", "-r",
        type=int,
        default=0,
        help="Which verification rollout to use as source (default: 0)",
    )
    force_parser.add_argument(
        "--num-forces", "-n",
        type=int,
        default=5,
        help="Number of force attempts per sentence (default: 5)",
    )
    force_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.8,
        help="Agreement threshold for selecting questions (default: 0.8)",
    )
    force_parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=300,
        help="Maximum concurrent API calls (default: 300)",
    )
    force_parser.set_defaults(func=cmd_force)

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
        "--threshold", "-t",
        type=float,
        default=0.8,
        help="Agreement threshold (default: 0.8)",
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
    full_parser.set_defaults(func=cmd_full)

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List verified questions and forcing results",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    # Check for API key
    if not args.api_key and not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: No API key provided. Set OPENROUTER_API_KEY or use --api-key")
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
