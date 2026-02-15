"""
Run the blackmail counterfactual dataset generation pipeline.

Usage:
    python -m src2.runs.run_blackmail_counterfactual --generate-counterfactuals
    python -m src2.runs.run_blackmail_counterfactual --run-rollouts
    python -m src2.runs.run_blackmail_counterfactual --judge
    python -m src2.runs.run_blackmail_counterfactual --build-csv
    python -m src2.runs.run_blackmail_counterfactual --all

Options:
    --subject-model MODEL   Subject model (default: moonshotai/kimi-k2-thinking)
    --num-samples N         Rollouts per condition (default: 50)
    --max-workers N         Thread pool size (default: 100)
    --max-conditions N      Limit number of counterfactual conditions (for testing)
    --num-per-feature N     Counterfactuals to generate per feature (default: 10)
"""

import argparse

from dotenv import load_dotenv

from src2.tasks.blackmail_counterfactual.task import BlackmailCounterfactualTask

load_dotenv()

DEFAULT_SUBJECT_MODEL = "moonshotai/kimi-k2-thinking"


def main():
    parser = argparse.ArgumentParser(
        description="Blackmail counterfactual dataset pipeline"
    )
    parser.add_argument(
        "--generate-counterfactuals",
        action="store_true",
        help="Generate counterfactual email modifications via LLM",
    )
    parser.add_argument(
        "--run-rollouts",
        action="store_true",
        help="Run model rollouts for all conditions",
    )
    parser.add_argument(
        "--judge", action="store_true", help="Judge all unjudged rollouts"
    )
    parser.add_argument(
        "--build-csv", action="store_true", help="Build CSV from rollout JSONs"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps in sequence",
    )
    parser.add_argument(
        "--subject-model",
        default=DEFAULT_SUBJECT_MODEL,
        help=f"Subject model (default: {DEFAULT_SUBJECT_MODEL})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of rollouts per condition (default: 50)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Thread pool size (default: 100)",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help="Limit number of counterfactual conditions (for testing)",
    )
    parser.add_argument(
        "--num-per-feature",
        type=int,
        default=10,
        help="Counterfactuals to generate per feature (default: 10)",
    )

    args = parser.parse_args()

    # Default to --all if no flags specified
    if not any([
        args.generate_counterfactuals, args.run_rollouts,
        args.judge, args.build_csv, args.all,
    ]):
        parser.print_help()
        return

    task = BlackmailCounterfactualTask(
        subject_model=args.subject_model,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
    )

    print(f"Task: {task.name}")
    print(f"Data dir: {task.data_dir}")
    print(f"Subject model: {task.subject_model}")
    print()

    if args.generate_counterfactuals or args.all:
        print("=" * 60)
        print("Step 1: Generating counterfactuals")
        print("=" * 60)
        task.generate_counterfactuals(num_per_feature=args.num_per_feature)
        print()

    if args.run_rollouts or args.all:
        print("=" * 60)
        print("Step 2: Running rollouts")
        print("=" * 60)
        task.run_data(
            max_conditions=args.max_conditions,
            num_samples=args.num_samples,
        )
        print()

    if args.judge or args.all:
        print("=" * 60)
        print("Step 3: Judging rollouts")
        print("=" * 60)
        task.judge_rollouts()
        print()

    if args.build_csv or args.all:
        print("=" * 60)
        print("Step 4: Building CSV")
        print("=" * 60)
        df = task.build_csv()
        print(f"\nSummary:")
        print(f"  Total rows: {len(df)}")
        if len(df) > 0:
            print(f"  Conditions: {df['condition_id'].nunique()}")
            print(f"  Blackmail (primary): {df['blackmail_primary'].sum()}")
            print(f"  Blackmail (at all): {df['blackmail_at_all'].sum()}")
            if "primary_category" in df.columns:
                print(f"\n  Category distribution:")
                for cat, count in df["primary_category"].value_counts().items():
                    print(f"    {cat}: {count}")
        print()


if __name__ == "__main__":
    main()
