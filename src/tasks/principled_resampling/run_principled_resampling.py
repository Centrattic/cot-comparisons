#!/usr/bin/env python3
"""
Main script for the Principled Resampling (Thought Branches CI++) task.

Usage:
    # Generate blackmail rollouts
    python run_principled_resampling.py rollouts --num-rollouts 50

    # Compute CI++ for one rollout
    python run_principled_resampling.py ci --rollout-idx 0 --num-resamples 100

    # List available rollouts and CI++ results
    python run_principled_resampling.py list

    # Print CI++ summary for a completed run
    python run_principled_resampling.py summarize --rollout-idx 0
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

DEFAULT_MODEL = "qwen/qwen3-32b"


def cmd_rollouts(args):
    """Generate blackmail rollouts with Qwen3-32B."""
    from src.tasks.principled_resampling.rollouts import generate_blackmail_rollouts

    print("=" * 60)
    print("PRINCIPLED RESAMPLING: Generate Blackmail Rollouts")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Num rollouts: {args.num_rollouts}")
    print()

    rollouts = asyncio.run(generate_blackmail_rollouts(
        num_rollouts=args.num_rollouts,
        model=args.model,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))

    blackmail_rollouts = [r for r in rollouts if r.get("is_blackmail")]
    print(f"\nDone. {len(blackmail_rollouts)} blackmail rollouts available for CI++ analysis.")


def cmd_ci(args):
    """Compute CI++ for one rollout."""
    from src.tasks.principled_resampling.task import PrincipledResamplingTask
    from src.tasks.principled_resampling.resampling import resample_sentence, ResampleResult
    from src.tasks.principled_resampling.similarity import (
        compute_similarity_matrix,
        compute_median_threshold,
    )
    from src.tasks.principled_resampling.metrics import compute_ci_and_ci_plus_plus
    from src.tasks.forced_response.prompts import split_cot_into_sentences
    from src.tasks.forced_response.data_loader import BinaryJudgeQuestion, load_custom_questions

    print("=" * 60)
    print("PRINCIPLED RESAMPLING: Compute CI++")
    print("=" * 60)

    task = PrincipledResamplingTask(model=args.model)

    # Load the source rollout
    rollout_dir = None
    if args.rollout_dir:
        rollout_dir = Path(args.rollout_dir)
    rollout = task.load_rollout(args.rollout_idx, run_dir=rollout_dir)
    if rollout is None:
        print(f"ERROR: No rollout found at index {args.rollout_idx}")
        print("Run 'rollouts' command first.")
        sys.exit(1)

    if not rollout.get("is_blackmail"):
        print(f"WARNING: Rollout {args.rollout_idx} was NOT judged as blackmail "
              f"(answer={rollout.get('answer')})")
        print("CI++ analysis is most meaningful on blackmail rollouts.")
        if not args.force:
            print("Use --force to proceed anyway.")
            sys.exit(1)

    # Extract and split CoT
    cot = rollout.get("thinking", "")
    if not cot:
        print("ERROR: Rollout has no thinking/CoT content.")
        sys.exit(1)

    sentences = split_cot_into_sentences(cot)
    print(f"Model: {args.model}")
    print(f"Rollout index: {args.rollout_idx}")
    print(f"CoT sentences: {len(sentences)}")
    print(f"Resamples per sentence: {args.num_resamples}")
    print(f"Total API calls: ~{len(sentences) * args.num_resamples}")
    print()

    # Load blackmail question for judging
    questions = load_custom_questions()
    question = None
    for q in questions:
        if isinstance(q, BinaryJudgeQuestion) and q.id == "blackmail_001":
            question = q
            break
    if question is None:
        print("ERROR: blackmail_001 question not found")
        sys.exit(1)

    # Create CI++ run dir
    config = {
        "model": args.model,
        "rollout_idx": args.rollout_idx,
        "num_resamples": args.num_resamples,
        "num_sentences": len(sentences),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_workers": args.max_workers,
        "source_cot": cot,
    }
    run_dir = task.create_ci_run_dir(args.rollout_idx, config)
    print(f"Saving to: {run_dir}")
    print()

    # Process all sentences in parallel
    async def process_sentence(i, sentence):
        """Resample, compute similarity, and compute CI/CI++ for one sentence."""
        # Check if this sentence was already computed (resume support)
        existing_summary = run_dir / f"sentence_{i:03d}" / "summary.json"
        if existing_summary.exists() and not args.recompute:
            print(f"  [Sentence {i}] Already computed, skipping (use --recompute to redo)")
            with open(existing_summary) as f:
                summary = json.load(f)
            return summary

        # Step 1: Resample
        print(f"  [Sentence {i}] Resampling {args.num_resamples} continuations...")
        resample_results = await resample_sentence(
            question=question,
            source_cot_sentences=sentences,
            sentence_idx=i,
            num_resamples=args.num_resamples,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            verbose=False,
        )

        # Convert to dicts for saving and metrics
        resample_dicts = [r.to_dict() for r in resample_results]

        # Save resamples
        task.save_sentence_resamples(i, resample_dicts, run_dir)

        # Step 2: Compute similarities
        continuation_sentences_list = [r.continuation_sentences for r in resample_results]
        similarities = compute_similarity_matrix(sentence, continuation_sentences_list)
        threshold = compute_median_threshold(similarities)

        similarity_data = {
            "source_sentence": sentence,
            "source_sentence_idx": i,
            "threshold": threshold,
            "similarities": similarities,
            "num_resamples": len(resample_results),
        }
        task.save_sentence_similarity(i, similarity_data, run_dir)

        # Step 3: Compute CI and CI++
        metrics = compute_ci_and_ci_plus_plus(
            resamples=resample_dicts,
            all_similarities=similarities,
            threshold=threshold,
        )

        # Add sentence info to metrics
        metrics["sentence_idx"] = i
        metrics["sentence_text"] = sentence

        task.save_sentence_summary(i, metrics, run_dir)

        # Print results for this sentence
        print(f"  [Sentence {i}] CI={metrics['ci']:.4f}  CI++={metrics['ci_plus_plus']:.4f}  "
              f"P(blackmail): baseline={metrics['p_blackmail_baseline']:.2f} "
              f"ci++={metrics['p_blackmail_ci_plus_plus']:.2f}  "
              f"resilience={metrics['resilience_mean']:.1f}  "
              f"filtered: {metrics['num_filtered_ci_plus_plus']}/{metrics['num_resamples_total']}")

        return metrics

    async def process_all_sentences():
        tasks = [process_sentence(i, sentence) for i, sentence in enumerate(sentences)]
        return await asyncio.gather(*tasks)

    print(f"Processing all {len(sentences)} sentences in parallel...")
    all_sentence_results = list(asyncio.run(process_all_sentences()))

    # Save aggregated results
    results = {
        "rollout_idx": args.rollout_idx,
        "model": args.model,
        "num_sentences": len(sentences),
        "num_resamples_per_sentence": args.num_resamples,
        "source_cot": cot,
        "sentences": all_sentence_results,
    }
    task.save_ci_results(results, run_dir)

    # Print summary
    print("=" * 60)
    print("CI++ RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Idx':>4} {'CI':>8} {'CI++':>8} {'P(bm)_base':>10} {'P(bm)_ci++':>10} {'Resil':>6} {'Filt':>6}  Sentence")
    print("-" * 100)
    for s in all_sentence_results:
        text = s.get("sentence_text", "")[:50]
        print(f"{s['sentence_idx']:4d} {s['ci']:8.4f} {s['ci_plus_plus']:8.4f} "
              f"{s['p_blackmail_baseline']:10.3f} {s['p_blackmail_ci_plus_plus']:10.3f} "
              f"{s['resilience_mean']:6.1f} {s['num_filtered_ci_plus_plus']:6d}  {text}")

    print(f"\nResults saved to: {run_dir / 'results.json'}")


def cmd_list(args):
    """List available rollouts and CI++ results."""
    from src.tasks.principled_resampling.task import PrincipledResamplingTask

    task = PrincipledResamplingTask()

    print("=" * 60)
    print("PRINCIPLED RESAMPLING: Available Data")
    print("=" * 60)

    # List rollout runs
    print("\nROLLOUT RUNS:")
    if task.rollouts_dir.exists():
        runs = sorted(d for d in task.rollouts_dir.iterdir() if d.is_dir())
        if runs:
            for run_dir in runs:
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    print(f"  {run_dir.name}: "
                          f"{summary['total_rollouts']} rollouts, "
                          f"{summary['blackmail_count']} blackmail "
                          f"({summary['blackmail_rate']:.0%})")
                    if summary.get("blackmail_rollout_indices"):
                        print(f"    Blackmail indices: {summary['blackmail_rollout_indices']}")
                else:
                    print(f"  {run_dir.name}: (no summary)")
        else:
            print("  (none)")
    else:
        print("  (none)")

    # List CI++ runs
    print("\nCI++ RUNS:")
    if task.ci_dir.exists():
        rollout_dirs = sorted(d for d in task.ci_dir.iterdir() if d.is_dir())
        if rollout_dirs:
            for rollout_dir in rollout_dirs:
                runs = sorted(d for d in rollout_dir.iterdir() if d.is_dir())
                for run_dir in runs:
                    results_path = run_dir / "results.json"
                    if results_path.exists():
                        with open(results_path) as f:
                            results = json.load(f)
                        n_sentences = results.get("num_sentences", "?")
                        print(f"  {rollout_dir.name}/{run_dir.name}: "
                              f"{n_sentences} sentences analyzed")
                    else:
                        # Check partial progress
                        sentence_dirs = sorted(
                            d for d in run_dir.iterdir()
                            if d.is_dir() and d.name.startswith("sentence_")
                        )
                        completed = sum(
                            1 for d in sentence_dirs
                            if (d / "summary.json").exists()
                        )
                        print(f"  {rollout_dir.name}/{run_dir.name}: "
                              f"{completed} sentences completed (in progress)")
        else:
            print("  (none)")
    else:
        print("  (none)")


def cmd_summarize(args):
    """Print CI++ summary for a completed run."""
    from src.tasks.principled_resampling.task import PrincipledResamplingTask

    task = PrincipledResamplingTask()
    results = task.load_ci_results(args.rollout_idx)

    if results is None:
        print(f"No CI++ results found for rollout {args.rollout_idx}")
        sys.exit(1)

    print("=" * 60)
    print(f"CI++ RESULTS: Rollout {args.rollout_idx}")
    print("=" * 60)
    print(f"Model: {results.get('model', 'unknown')}")
    print(f"Sentences: {results['num_sentences']}")
    print(f"Resamples per sentence: {results.get('num_resamples_per_sentence', '?')}")
    print()

    print(f"{'Idx':>4} {'CI':>8} {'CI++':>8} {'P(bm)_base':>10} {'P(bm)_ci++':>10} {'Resil':>6} {'Filt':>6}  Sentence")
    print("-" * 100)

    for s in results["sentences"]:
        text = s.get("sentence_text", "")[:50]
        print(f"{s['sentence_idx']:4d} {s['ci']:8.4f} {s['ci_plus_plus']:8.4f} "
              f"{s['p_blackmail_baseline']:10.3f} {s['p_blackmail_ci_plus_plus']:10.3f} "
              f"{s['resilience_mean']:6.1f} {s['num_filtered_ci_plus_plus']:6d}  {text}")

    # Highlight top-CI++ sentences
    sorted_by_cipp = sorted(results["sentences"], key=lambda x: x["ci_plus_plus"], reverse=True)
    print(f"\nTop 5 sentences by CI++:")
    for s in sorted_by_cipp[:5]:
        text = s.get("sentence_text", "")
        print(f"  [{s['sentence_idx']}] CI++={s['ci_plus_plus']:.4f}: {text[:100]}")


def main():
    parser = argparse.ArgumentParser(
        description="Principled Resampling (Thought Branches CI++) Task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # rollouts
    p_rollouts = subparsers.add_parser("rollouts", help="Generate blackmail rollouts")
    p_rollouts.add_argument("--num-rollouts", type=int, default=50)
    p_rollouts.add_argument("--max-workers", type=int, default=50)
    p_rollouts.add_argument("--temperature", type=float, default=0.7)
    p_rollouts.add_argument("--max-tokens", type=int, default=16000)

    # ci
    p_ci = subparsers.add_parser("ci", help="Compute CI++ for one rollout")
    p_ci.add_argument("--rollout-idx", type=int, default=0, help="Index of rollout to analyze")
    p_ci.add_argument("--rollout-dir", type=str, default=None, help="Specific rollout run dir")
    p_ci.add_argument("--num-resamples", type=int, default=100)
    p_ci.add_argument("--max-workers", type=int, default=50)
    p_ci.add_argument("--temperature", type=float, default=0.7)
    p_ci.add_argument("--max-tokens", type=int, default=16000)
    p_ci.add_argument("--force", action="store_true", help="Proceed even if rollout isn't blackmail")
    p_ci.add_argument("--recompute", action="store_true", help="Recompute already-completed sentences")

    # list
    subparsers.add_parser("list", help="List available data")

    # summarize
    p_summarize = subparsers.add_parser("summarize", help="Print CI++ results")
    p_summarize.add_argument("--rollout-idx", type=int, default=0)

    args = parser.parse_args()

    commands = {
        "rollouts": cmd_rollouts,
        "ci": cmd_ci,
        "list": cmd_list,
        "summarize": cmd_summarize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
