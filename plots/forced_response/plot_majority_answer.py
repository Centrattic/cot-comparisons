"""
Plot majority answer over sentence index for forcing, monitor, and resampling.

Usage:
    python plots/forced_response/plot_majority_answer.py -q custom_bagel_001 -r 0
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "forced_response"
PLOT_DIR = Path(__file__).parent

ANSWER_LABELS = ['A', 'B', 'C', 'D']
ANSWER_TO_Y = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def get_latest_run_dir(rollout_dir: Path):
    """Get the latest timestamped run dir, or rollout_dir itself if legacy."""
    if not rollout_dir.exists():
        return None
    timestamped = sorted(
        [d for d in rollout_dir.iterdir()
         if d.is_dir() and len(d.name) >= 15 and d.name[8] == '_'],
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    if (rollout_dir / "summary.json").exists():
        return rollout_dir
    return None


def load_forcing(question_id: str, rollout_idx: int):
    rollout_dir = DATA_DIR / "forcing" / question_id / f"rollout_{rollout_idx:03d}"
    run_dir = get_latest_run_dir(rollout_dir)
    if not run_dir:
        return None, None
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None, None
    with open(summary_path) as f:
        data = json.load(f)
    indices = []
    answers = []
    for s in data['sentence_summaries']:
        most = s.get('most_common', '')
        if most in ANSWER_TO_Y:
            indices.append(s['sentence_idx'])
            answers.append(ANSWER_TO_Y[most])
    return indices, answers


def load_monitor_forcing(question_id: str, rollout_idx: int):
    rollout_dir = DATA_DIR / "monitor_forcing" / question_id / f"rollout_{rollout_idx:03d}"
    run_dir = get_latest_run_dir(rollout_dir)
    if not run_dir:
        return None, None
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None, None
    with open(summary_path) as f:
        data = json.load(f)
    indices = []
    answers = []
    for s in data['sentence_summaries']:
        most = s.get('most_common', '')
        if most in ANSWER_TO_Y:
            indices.append(s['sentence_idx'])
            answers.append(ANSWER_TO_Y[most])
    return indices, answers


def load_monitor_resampling(question_id: str, rollout_idx: int):
    rollout_dir = DATA_DIR / "monitor_resampling" / question_id / f"rollout_{rollout_idx:03d}"
    run_dir = get_latest_run_dir(rollout_dir)
    if not run_dir:
        return None, None
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None, None
    with open(summary_path) as f:
        data = json.load(f)
    indices = []
    answers = []
    for s in data['sentence_summaries']:
        most = s.get('most_common', '')
        if most in ANSWER_TO_Y:
            indices.append(s['sentence_idx'])
            answers.append(ANSWER_TO_Y[most])
    return indices, answers


def load_resampling(question_id: str, rollout_idx: int):
    rollout_dir = DATA_DIR / "resampling" / question_id / f"rollout_{rollout_idx:03d}"
    run_dir = get_latest_run_dir(rollout_dir)
    if not run_dir:
        return None, None
    indices = []
    answers = []
    for sentence_dir in sorted(run_dir.iterdir()):
        if not sentence_dir.is_dir() or not sentence_dir.name.startswith("sentence_"):
            continue
        summary_path = sentence_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)
        most = data.get('most_common', '')
        if most in ANSWER_TO_Y:
            idx = int(sentence_dir.name.split("_")[1])
            indices.append(idx)
            answers.append(ANSWER_TO_Y[most])
    return indices, answers


def plot_majority_answers(question_id: str, rollout_idx: int = 0):
    force_x, force_y = load_forcing(question_id, rollout_idx)
    mf_x, mf_y = load_monitor_forcing(question_id, rollout_idx)
    resample_x, resample_y = load_resampling(question_id, rollout_idx)
    mr_x, mr_y = load_monitor_resampling(question_id, rollout_idx)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot each series
    if resample_x:
        ax.scatter(resample_x, resample_y, marker='D', s=50, color='#2ca02c',
                   zorder=5, label='Resampling')
        ax.plot(resample_x, resample_y, color='#2ca02c', alpha=0.4, linewidth=1.5)

    if mr_x:
        ax.scatter(mr_x, mr_y, marker='s', s=40, color='#98df8a',
                   zorder=4, label='Monitor (Resampling)')
        ax.plot(mr_x, mr_y, color='#98df8a', alpha=0.5, linewidth=1.2)

    if force_x:
        ax.scatter(force_x, force_y, marker='o', s=25, color='#1f77b4',
                   zorder=3, label='Forcing (Tinker)')
        ax.plot(force_x, force_y, color='#1f77b4', alpha=0.5, linewidth=1.2)

    if mf_x:
        ax.scatter(mf_x, mf_y, marker='x', s=25, color='#d62728',
                   zorder=2, label='Monitor (Forcing)')
        ax.plot(mf_x, mf_y, color='#d62728', alpha=0.4, linewidth=1.2)

    # Load correct answer from any available summary
    correct = None
    for mode in ["forcing", "monitor_forcing", "resampling", "monitor_resampling"]:
        rollout_dir = DATA_DIR / mode / question_id / f"rollout_{rollout_idx:03d}"
        run_dir = get_latest_run_dir(rollout_dir)
        if run_dir:
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    data = json.load(f)
                correct = data.get('correct_answer', '')
                if correct:
                    break

    if correct and correct in ANSWER_TO_Y:
        ax.axhline(y=ANSWER_TO_Y[correct], color='green', linestyle='--',
                   alpha=0.3, linewidth=2, label=f'Correct ({correct})')

    # Formatting
    ax.set_yticks(range(4))
    ax.set_yticklabels(ANSWER_LABELS)
    ax.set_xlabel('Sentence Index')
    ax.set_ylabel('Majority Answer')
    ax.set_title(f'Majority Answer by Sentence — {question_id} (rollout {rollout_idx})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    all_x = (force_x or [0])[-1:] + (mf_x or [0])[-1:] + (resample_x or [0])[-1:] + (mr_x or [0])[-1:]
    ax.set_xlim(-1, max(all_x) + 2)

    plt.tight_layout()
    out_path = PLOT_DIR / f"majority_answer_{question_id}_rollout{rollout_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot majority answer comparison")
    parser.add_argument("-q", "--question-id", required=True)
    parser.add_argument("-r", "--rollout-idx", type=int, default=0)
    args = parser.parse_args()

    plot_majority_answers(args.question_id, args.rollout_idx)
