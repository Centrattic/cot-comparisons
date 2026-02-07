#!/usr/bin/env python3
"""
Forced-answer entropy analysis for a single scruples prompt.

Picks control (non-sycophantic) and intervention (sycophantic) rollouts,
forces the model to answer at each sentence boundary, and plots
entropy of the forced answer distribution vs. fraction of reasoning remaining.

Usage:
    source ~/2025_env/bin/activate
    cd /home/riya/neel-projs/cot-comparisons
    python -m src2.runs.run_scruples_forcing
"""

import argparse
import contextlib
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm

from src2.tasks.forced_response.prompts import (
    get_cumulative_cot_segments,
    split_cot_into_sentences,
)
from src2.utils.chat_template import build_thinking_prompt

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scruples"
OUTPUT_DIR = DATA_DIR / "forced_response"

SUBJECT_MODEL = "Qwen/Qwen3-32B"
ANECDOTE_ID = "iazYMdUuXZTkCMStPYzs0hew2gWG4FJI"  # phone call bus trip

CHOICES = ["A", "B"]
TOPK = 20
MAX_CONTROL_ROLLOUTS = 5
MAX_INTERVENTION_ROLLOUTS = 5
SENTENCE_STRIDE = 1  # use every sentence


# ── Helpers ───────────────────────────────────────────────────────────

def extract_thinking_text(thinking_field) -> str:
    """Extract thinking text from the run JSON thinking field."""
    if isinstance(thinking_field, str):
        return thinking_field
    if isinstance(thinking_field, list):
        parts = []
        for item in thinking_field:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(thinking_field)


def shannon_entropy(probs: Dict[str, float]) -> float:
    """Compute Shannon entropy in bits from a probability distribution."""
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def resolve_choice_token_ids(tokenizer, choices: List[str]) -> Dict[str, int]:
    """Map each answer string to its single token id."""
    mapping = {}
    for c in choices:
        with contextlib.redirect_stdout(io.StringIO()):
            ids = tokenizer.encode(c, add_special_tokens=False)
        mapping[c] = ids[-1]
    return mapping


def get_choice_logprobs_tinker(
    sampling_client,
    tokenizer,
    prompt_str: str,
    choices: List[str],
    choice_token_ids: Dict[str, int],
    types_module,
    topk: int = TOPK,
) -> Dict[str, float]:
    """
    Get forced answer distribution using Tinker's topk_prompt_logprobs.
    Returns normalized probability distribution over choices.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

    # Append dummy token so "last prompt position" = generation slot
    dummy_id = choice_token_ids[choices[0]]
    extended_tokens = prompt_tokens + [dummy_id]
    extended_input = types_module.ModelInput.from_ints(extended_tokens)

    topk_result = sampling_client.sample(
        prompt=extended_input,
        num_samples=1,
        sampling_params=types_module.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=topk,
    ).result()

    topk_at_gen = topk_result.topk_prompt_logprobs[-1] if topk_result.topk_prompt_logprobs else []
    topk_lookup = {tid: lp for tid, lp in topk_at_gen} if topk_at_gen else {}

    # Extract logprobs for answer choices
    choice_logprobs = {}
    for c in choices:
        tid = choice_token_ids[c]
        choice_logprobs[c] = topk_lookup.get(tid, None)

    # Softmax over found choices
    found = {c: lp for c, lp in choice_logprobs.items() if lp is not None}
    if found:
        max_lp = max(found.values())
        exps = {c: math.exp(lp - max_lp) for c, lp in found.items()}
        total = sum(exps.values())
        choice_probs = {c: exps.get(c, 0.0) / total for c in choices}
    else:
        choice_probs = {c: 1.0 / len(choices) for c in choices}

    return choice_probs


def force_rollout(
    sampling_client,
    tokenizer,
    types_module,
    user_msg: str,
    thinking_text: str,
    choices: List[str],
    choice_token_ids: Dict[str, int],
    sentence_stride: int = 1,
) -> List[Dict]:
    """
    Force answer at each sentence boundary of a CoT rollout.
    Returns list of {sentence_idx, fraction_remaining, choice_probs, entropy}.
    """
    segments = get_cumulative_cot_segments(thinking_text)
    num_sentences = len(segments)

    if num_sentences == 0:
        return []

    # Apply stride
    if sentence_stride > 1:
        indices = list(range(0, num_sentences, sentence_stride))
        if (num_sentences - 1) not in indices:
            indices.append(num_sentences - 1)
    else:
        indices = list(range(num_sentences))

    results = []

    def _force_at(idx):
        partial_cot = segments[idx]
        anchor = " So, the answer is: " if partial_cot else "So, the answer is: "
        cot_with_anchor = partial_cot + anchor
        prompt_str = build_thinking_prompt(
            tokenizer, user_msg, cot_prefix=cot_with_anchor,
        ) + "</think>\n"

        probs = get_choice_logprobs_tinker(
            sampling_client, tokenizer, prompt_str,
            choices, choice_token_ids, types_module,
        )
        fraction_remaining = 1.0 - (idx + 1) / num_sentences
        entropy = shannon_entropy(probs)
        return {
            "sentence_idx": idx,
            "fraction_remaining": fraction_remaining,
            "choice_probs": probs,
            "entropy": entropy,
        }

    # Parallelize forcing calls
    with ThreadPoolExecutor(max_workers=min(100, len(indices))) as executor:
        futures = {executor.submit(_force_at, idx): idx for idx in indices}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["sentence_idx"])
    return results


# ── Load rollouts ────────────────────────────────────────────────────

def load_rollouts(anecdote_id: str, arm: str, max_rollouts: int) -> List[Dict]:
    """Load rollout JSONs for one anecdote from the results CSV."""
    results_csv = DATA_DIR / "results_suggest_wrong.csv"
    df = pd.read_csv(results_csv)
    rows = df[(df["anecdote_id"] == anecdote_id) & (df["arm"] == arm)]

    # For intervention, prefer sycophantic ones; for control, prefer non-sycophantic
    if arm == "intervention":
        rows = rows[rows["is_sycophantic"] == True]
    else:
        rows = rows[rows["is_sycophantic"] == False]

    rows = rows.head(max_rollouts)
    rollouts = []
    for _, row in rows.iterrows():
        run_path = DATA_DIR / row["run_path"]
        if run_path.exists():
            with open(run_path) as f:
                data = json.load(f)
            thinking = extract_thinking_text(data.get("thinking", ""))
            if len(thinking) > 100:  # skip empty/tiny CoTs
                rollouts.append({
                    "run_idx": row["run_idx"],
                    "arm": arm,
                    "answer": row["answer"],
                    "is_sycophantic": row["is_sycophantic"],
                    "user_msg": data["prompt"],
                    "thinking": thinking,
                })
    return rollouts


# ── Plotting ─────────────────────────────────────────────────────────

def plot_entropy(
    control_results: List[Tuple[int, List[Dict]]],
    intervention_results: List[Tuple[int, List[Dict]]],
    title: str,
    save_path: Path,
    ctrl_label: str = "Control (non-sycophantic)",
    intv_label: str = "Intervention (sycophantic)",
    suptitle: Optional[str] = None,
):
    """
    Create entropy vs fraction of reasoning remaining plot.

    Two subplots: control (left) and intervention (right).
    Points colored by most likely answer (A=red/sycophantic, B=blue/non-sycophantic).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    max_entropy = math.log2(len(CHOICES))  # 1.0 bit for binary

    COLOR_A = "#d62728"  # red — sycophantic answer
    COLOR_B = "#1f77b4"  # blue — non-sycophantic answer

    for ax, results_list, arm_label in [
        (axes[0], control_results, ctrl_label),
        (axes[1], intervention_results, intv_label),
    ]:
        all_frac = []
        all_ent = []
        all_colors = []

        for rollout_idx, results in results_list:
            for r in results:
                all_frac.append(r["fraction_remaining"])
                all_ent.append(r["entropy"])
                # Color by most likely answer
                argmax = max(r["choice_probs"], key=r["choice_probs"].get)
                all_colors.append(COLOR_A if argmax == "A" else COLOR_B)

        all_frac = np.array(all_frac)
        all_ent = np.array(all_ent)

        # Scatter colored by argmax answer
        sc = ax.scatter(
            all_frac, all_ent, c=all_colors,
            alpha=0.7, s=40, edgecolors="none",
        )

        # Linear fit
        if len(all_frac) > 2:
            slope, intercept, r_val, p_val, std_err = stats.linregress(all_frac, all_ent)
            x_fit = np.linspace(0, 1, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "r--", alpha=0.8, linewidth=1.5)

            # r value annotation
            ax.text(
                0.98, 0.95, f"r={r_val:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Max entropy line
        ax.axhline(
            y=max_entropy, color="gray", linestyle=":", alpha=0.5, linewidth=1,
        )
        ax.text(
            0.02, max_entropy + 0.02, "max (uniform)",
            transform=ax.get_xaxis_transform(),
            fontsize=8, color="gray", alpha=0.7,
        )

        # Legend for answer colors
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_A,
                   markersize=8, label="A (sycophantic)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_B,
                   markersize=8, label="B (non-sycophantic)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
                  framealpha=0.8)

        n_rollouts = len(results_list)
        total_sents = sum(len(r) for _, r in results_list)
        n_a = sum(1 for c in all_colors if c == COLOR_A)
        n_b = sum(1 for c in all_colors if c == COLOR_B)
        ax.set_title(f"{arm_label}\n({n_rollouts} rollouts, {total_sents} pts — {n_a} A, {n_b} B)", fontsize=11)
        ax.set_xlabel("Fraction of reasoning remaining", fontsize=10)
        ax.set_xlim(1.05, -0.05)  # reversed x axis
        ax.set_ylim(-0.05, max_entropy + 0.15)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Entropy (bits)", fontsize=10)

    fig.suptitle(
        suptitle or f"Forced Answer Entropy — Scruples\n"
        f"\"AITA? Phone for 3h on 10h bus trip\" (answers: A/B, sycophantic=A)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── Replot from JSON ─────────────────────────────────────────────────

def replot_from_json(
    json_path: Path,
    save_path: Path,
    suptitle: str,
    ctrl_label: str = "Control (non-sycophantic only)",
    intv_label: str = "Intervention (sycophantic only)",
):
    """Replot from saved JSON, filtering to matching sycophancy status."""
    with open(json_path) as f:
        data = json.load(f)

    # Filter control to non-sycophantic, intervention to sycophantic
    control_results = []
    ctrl_skipped = 0
    for entry in data["control"]:
        if not entry["is_sycophantic"]:
            control_results.append((entry["run_idx"], entry["results"]))
        else:
            ctrl_skipped += 1

    intervention_results = []
    intv_skipped = 0
    for entry in data["intervention"]:
        if entry["is_sycophantic"]:
            intervention_results.append((entry["run_idx"], entry["results"]))
        else:
            intv_skipped += 1

    print(f"Control: kept {len(control_results)}, filtered out {ctrl_skipped} sycophantic")
    print(f"Intervention: kept {len(intervention_results)}, filtered out {intv_skipped} non-sycophantic")

    if not control_results or not intervention_results:
        print("ERROR: No rollouts left after filtering!")
        return

    plot_entropy(
        control_results, intervention_results,
        title="replot", save_path=save_path,
        ctrl_label=ctrl_label, intv_label=intv_label,
        suptitle=suptitle,
    )


# ── Main ─────────────────────────────────────────────────────────────

def print_summary(label: str, results_list: List[Tuple[int, List[Dict]]]):
    """Print summary stats for one arm."""
    all_ent = [r["entropy"] for _, results in results_list for r in results]
    early = [r["entropy"] for _, results in results_list for r in results if r["fraction_remaining"] > 0.7]
    late = [r["entropy"] for _, results in results_list for r in results if r["fraction_remaining"] < 0.3]
    print(f"  {label}:")
    print(f"    Mean entropy: {np.mean(all_ent):.3f} bits")
    print(f"    Early (>70% remaining): {np.mean(early):.3f} bits" if early else "    Early: N/A")
    print(f"    Late (<30% remaining): {np.mean(late):.3f} bits" if late else "    Late: N/A")


def cmd_force(args):
    """Run forcing on rollouts (requires Tinker)."""
    from tinker import ServiceClient, types
    from transformers import AutoTokenizer

    anecdote_id = args.anecdote_id
    suffix = f"_{args.suffix}" if args.suffix else ""

    print("=" * 70)
    print("Scruples Forced-Answer Entropy Analysis")
    print(f"Anecdote: {anecdote_id}")
    print("=" * 70)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, trust_remote_code=True)
    client = ServiceClient()
    sampling_client = client.create_sampling_client(base_model=SUBJECT_MODEL)
    choice_token_ids = resolve_choice_token_ids(tokenizer, CHOICES)
    print(f"Choice token IDs: {choice_token_ids}")

    # Load rollouts (filtering by sycophancy status happens in load_rollouts)
    print(f"\nLoading rollouts for anecdote {anecdote_id}...")
    control_rollouts = load_rollouts(anecdote_id, "control", MAX_CONTROL_ROLLOUTS)
    intervention_rollouts = load_rollouts(anecdote_id, "intervention", MAX_INTERVENTION_ROLLOUTS)
    print(f"  Control rollouts: {len(control_rollouts)} (all non-sycophantic)")
    print(f"  Intervention rollouts: {len(intervention_rollouts)} (all sycophantic)")

    if not control_rollouts or not intervention_rollouts:
        print("ERROR: Not enough rollouts found!")
        return

    # Force each rollout
    all_control = []
    all_intervention = []

    for rollout in tqdm(control_rollouts, desc="Forcing control rollouts"):
        results = force_rollout(
            sampling_client, tokenizer, types,
            rollout["user_msg"], rollout["thinking"],
            CHOICES, choice_token_ids, SENTENCE_STRIDE,
        )
        all_control.append((rollout["run_idx"], results))
        n_sent = len(split_cot_into_sentences(rollout["thinking"]))
        print(f"  Control run {rollout['run_idx']}: {n_sent} sentences, answer={rollout['answer']}")

    for rollout in tqdm(intervention_rollouts, desc="Forcing intervention rollouts"):
        results = force_rollout(
            sampling_client, tokenizer, types,
            rollout["user_msg"], rollout["thinking"],
            CHOICES, choice_token_ids, SENTENCE_STRIDE,
        )
        all_intervention.append((rollout["run_idx"], results))
        n_sent = len(split_cot_into_sentences(rollout["thinking"]))
        print(f"  Intervention run {rollout['run_idx']}: {n_sent} sentences, answer={rollout['answer']}")

    # Save raw results (include sycophancy metadata for replotting)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUT_DIR / f"forcing_results{suffix}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "anecdote_id": anecdote_id,
            "model": SUBJECT_MODEL,
            "choices": CHOICES,
            "timestamp": datetime.now().isoformat(),
            "control": [
                {"run_idx": ri, "answer": ro["answer"], "is_sycophantic": ro["is_sycophantic"], "results": res}
                for (ri, res), ro in zip(all_control, control_rollouts)
            ],
            "intervention": [
                {"run_idx": ri, "answer": ro["answer"], "is_sycophantic": ro["is_sycophantic"], "results": res}
                for (ri, res), ro in zip(all_intervention, intervention_rollouts)
            ],
        }, f, indent=2)
    print(f"\nRaw results saved: {results_path}")

    # Plot
    suptitle = args.title or (
        f"Forced Answer Entropy — Scruples\n"
        f"Anecdote {anecdote_id[:12]}... (answers: A/B, sycophantic=A)"
    )
    plot_path = OUTPUT_DIR / f"entropy_vs_remaining_scruples{suffix}.png"
    plot_entropy(
        all_control, all_intervention,
        title="Scruples Forcing", save_path=plot_path,
        suptitle=suptitle,
    )

    # Print summary stats
    print("\n" + "=" * 70)
    print("Summary:")
    print_summary("Control", all_control)
    print_summary("Intervention", all_intervention)


def cmd_replot(args):
    """Replot from existing JSON with proper sycophancy filtering."""
    json_path = Path(args.json)
    suffix = f"_{args.suffix}" if args.suffix else ""
    save_path = Path(args.output) if args.output else OUTPUT_DIR / f"entropy_vs_remaining_scruples{suffix}.png"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suptitle = args.title or "Forced Answer Entropy — Scruples (replotted with filtering)"

    print("=" * 70)
    print("Replotting with sycophancy filtering")
    print(f"Source: {json_path}")
    print(f"Output: {save_path}")
    print("=" * 70)

    replot_from_json(
        json_path, save_path=save_path, suptitle=suptitle,
    )


def main():
    parser = argparse.ArgumentParser(description="Scruples forced-answer entropy analysis")
    subparsers = parser.add_subparsers(dest="command")

    # Force subcommand (requires Tinker)
    p_force = subparsers.add_parser("force", help="Run forcing on rollouts (requires Tinker)")
    p_force.add_argument("--anecdote-id", default=ANECDOTE_ID, help="Anecdote ID to analyze")
    p_force.add_argument("--suffix", default="", help="Output file suffix (e.g. 'highctrl')")
    p_force.add_argument("--title", default=None, help="Custom plot suptitle")

    # Replot subcommand (no Tinker needed)
    p_replot = subparsers.add_parser("replot", help="Replot from saved JSON with proper filtering")
    p_replot.add_argument("json", help="Path to forcing results JSON")
    p_replot.add_argument("--suffix", default="", help="Output file suffix")
    p_replot.add_argument("--output", default=None, help="Output PNG path (overrides suffix)")
    p_replot.add_argument("--title", default=None, help="Custom plot suptitle")

    args = parser.parse_args()

    if args.command == "force":
        cmd_force(args)
    elif args.command == "replot":
        cmd_replot(args)
    else:
        # Default: legacy behavior (force with default anecdote)
        parser.print_help()


if __name__ == "__main__":
    main()
