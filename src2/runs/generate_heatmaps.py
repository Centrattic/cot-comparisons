"""
Unified CLI for generating pairwise similarity heatmaps from reasoning rollouts
or arbitrary text files.

Supports four heatmap types:
  - sentence_layer44   : layer-44 activation cosine similarity (sentence-level)
  - sentence_semantic   : semantic embedding cosine similarity (sentence-level)
  - paragraph_layer44  : layer-44 activation cosine similarity (paragraph-level)
  - paragraph_semantic  : semantic embedding cosine similarity (paragraph-level)

Usage examples:
  python -m src2.runs.generate_heatmaps --rollout data/.../bagel/rollout_5.json
  python -m src2.runs.generate_heatmaps --prompt bagel well -n 3
  python -m src2.runs.generate_heatmaps --prompt starfish -n 1 --types sentence_semantic
  python -m src2.runs.generate_heatmaps --rollout-dir data/.../unlabeled/bagel
  python -m src2.runs.generate_heatmaps --text pragmatic --types sentence_semantic
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from sklearn.metrics.pairwise import cosine_similarity

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "data/reasoning_evals/heatmaps_2"
DEFAULT_ACTIVATIONS_DIR = "data/reasoning_evals/heatmap_activations"
DEFAULT_ROLLOUTS_DIR = "data/reasoning_evals/rollouts/unlabeled"
DEFAULT_TEXT_DIR = "data/texts"
DEFAULT_POD = "mats_9"
DEFAULT_TYPES = ["sentence_layer44", "sentence_semantic"]
ALL_TYPES = [
    "sentence_layer44", "sentence_semantic",
    "paragraph_layer44", "paragraph_semantic",
]


# ── Sentence / paragraph splitting ───────────────────────────────────
def split_sentences(text):
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    tok = PunktSentenceTokenizer()
    spans = list(tok.span_tokenize(text))
    return [text[a:b] for a, b in spans]


def split_paragraphs(text):
    paras = []
    for m in re.finditer(r'(?s).+?(?=\n\n|\Z)', text):
        t = m.group().strip()
        if t:
            paras.append(t)
    return paras


# ── Plotting ─────────────────────────────────────────────────────────
def plot_png(sim_matrix, title, out_path, gran_label="Sentence"):
    n = sim_matrix.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    masked = np.where(mask, np.nan, sim_matrix)

    valid = masked[~np.isnan(masked)]
    if len(valid) == 0:
        vmin, vmax = -1, 1
    else:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        if vmin < 0 < vmax:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound

    fig_size = max(8, min(14, n * 0.25))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(masked, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_xlabel(f"{gran_label} index")
    ax.set_ylabel(f"{gran_label} index")
    ax.set_title(title, fontsize=13, pad=12)
    ax.invert_yaxis()
    ax.yaxis.set_ticks_position("both")

    tick_step = max(1, n // 15)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Cosine similarity", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


def plot_html(sim_matrix, title, out_path, texts, gran_label="Sentence"):
    if not HAS_PLOTLY:
        return
    n = sim_matrix.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    masked = np.where(mask, np.nan, sim_matrix)

    valid = masked[~np.isnan(masked)]
    if len(valid) == 0:
        zmin, zmax = -1, 1
    else:
        zmin = float(np.percentile(valid, 2))
        zmax = float(np.percentile(valid, 98))
        if zmin < 0 < zmax:
            bound = max(abs(zmin), abs(zmax))
            zmin, zmax = -bound, bound

    wrap_width = 80
    hovertext = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                hovertext[i][j] = ""
            else:
                si = "<br>".join(textwrap.wrap(texts[i], wrap_width))
                sj = "<br>".join(textwrap.wrap(texts[j], wrap_width))
                hovertext[i][j] = (
                    f"<b>Similarity: {sim_matrix[i, j]:.3f}</b><br><br>"
                    f"<b>[{i}]</b> {si}<br><br>"
                    f"<b>[{j}]</b> {sj}"
                )

    fig = go.Figure(data=go.Heatmap(
        z=masked, hovertext=hovertext, hoverinfo="text",
        colorscale="RdBu_r", zmin=zmin, zmax=zmax,
        colorbar=dict(title="Cosine sim"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=f"{gran_label} index",
        yaxis_title=f"{gran_label} index",
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1,
                   side="right"),
        width=900, height=850,
    )
    fig.write_html(out_path)
    print(f"    Saved: {out_path}")


# ── Phase 1: Resolve inputs ──────────────────────────────────────────
def _make_stem(name, rollout_idx):
    """Build the file stem: 'name_rollout_N' for rollouts, 'name' for text."""
    if rollout_idx is None:
        return name
    return f"{name}_rollout_{rollout_idx}"


def resolve_inputs(args):
    """Return list of (name, rollout_idx, path) tuples.

    rollout_idx is None for --text inputs (plain text files).
    For rollout inputs, rollout_idx is an int.
    """
    items = []

    # --rollout: explicit rollout JSON paths
    if args.rollout:
        for path in args.rollout:
            with open(path) as f:
                data = json.load(f)
            items.append((
                data["prompt_name"],
                data["rollout_idx"],
                os.path.abspath(path),
            ))

    # --rollout-dir: all JSONs in directory
    if args.rollout_dir:
        for d in args.rollout_dir:
            for path in sorted(glob.glob(os.path.join(d, "*.json"))):
                with open(path) as f:
                    data = json.load(f)
                items.append((
                    data["prompt_name"],
                    data["rollout_idx"],
                    os.path.abspath(path),
                ))

    # --prompt: look up by name, generate if needed
    if args.prompt:
        for name in args.prompt:
            prompt_dir = os.path.join(DEFAULT_ROLLOUTS_DIR, name)
            existing = sorted(glob.glob(os.path.join(prompt_dir, "rollout_*.json")))
            if len(existing) < args.n:
                generate_missing_rollouts(name, args.n)
                existing = sorted(glob.glob(os.path.join(prompt_dir, "rollout_*.json")))
            for path in existing[:args.n]:
                with open(path) as f:
                    data = json.load(f)
                items.append((
                    data["prompt_name"],
                    data["rollout_idx"],
                    os.path.abspath(path),
                ))

    # --text: names resolved to data/texts/{name}.txt (rollout_idx=None)
    if args.text:
        for name in args.text:
            path = os.path.join(DEFAULT_TEXT_DIR, f"{name}.txt")
            if not os.path.exists(path):
                print(f"  ERROR: Text file not found: {path}")
                sys.exit(1)
            items.append((name, None, os.path.abspath(path)))

    return items


# ── Phase 2: Generate missing rollouts ───────────────────────────────
def generate_missing_rollouts(prompt_name, num_rollouts):
    """Use ReasoningEvalsTask to generate rollouts via Tinker."""
    print(f"\n  Generating rollouts for '{prompt_name}' (up to {num_rollouts})...")
    from ..tasks.reasoning_evals.task import ReasoningEvalsTask
    task = ReasoningEvalsTask(subject_model="Qwen/Qwen3-32B")
    task.generate_rollouts(
        prompt_names=[prompt_name],
        num_rollouts=num_rollouts,
        max_tokens=8192,
        temperature=0.7,
        workers=400,
    )


# ── Phase 3: Extract activations on pod ──────────────────────────────
def _make_rollout_json_for_text(text_path, name, tmp_dir):
    """Create a temporary rollout JSON from a plain text file."""
    with open(text_path) as f:
        text = f.read()
    rollout = {
        "prompt_name": name,
        "prompt_text": "Analyze the following text.",
        "rollout_idx": 0,
        "chain_of_thought": text,
        "output": "",
        "model": "text_input",
    }
    out_path = os.path.join(tmp_dir, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(rollout, f, indent=2)
    return out_path


def extract_activations_on_pod(items, activations_dir, pod):
    """SCP rollouts/text to pod, run extraction, SCP results back."""
    # Determine which items need extraction
    need_extraction = []
    for name, rollout_idx, path in items:
        stem = _make_stem(name, rollout_idx)
        npz_path = os.path.join(activations_dir, f"{stem}_activations.npz")
        manifest_path = os.path.join(activations_dir, f"{stem}_manifest.json")
        if not os.path.exists(npz_path) or not os.path.exists(manifest_path):
            need_extraction.append((name, rollout_idx, path, stem))

    if not need_extraction:
        print("\n  All activation files already exist, skipping pod extraction.")
        return

    print(f"\n  Extracting activations for {len(need_extraction)} rollout(s) on {pod}...")

    # Use absolute path on pod (~ not expanded inside Python strings)
    remote_home_result = subprocess.run(
        ["ssh", pod, "echo $HOME"], capture_output=True, text=True,
    )
    if remote_home_result.returncode != 0:
        print(f"    ERROR: Cannot reach pod '{pod}': {remote_home_result.stderr.strip()}")
        return
    remote_home = remote_home_result.stdout.strip()
    remote_base = f"{remote_home}/heatmap_extraction"
    remote_rollouts = f"{remote_base}/rollouts"
    remote_outputs = f"{remote_base}/outputs"

    def run_ssh(cmd, stream=False):
        full = ["ssh", pod, cmd]
        print(f"    $ ssh {pod} {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
        if stream:
            result = subprocess.run(full, text=True)
        else:
            result = subprocess.run(full, capture_output=True, text=True)
        if result.returncode != 0 and not stream:
            print(f"    STDERR: {result.stderr.strip()}")
        return result

    def run_scp(src, dst):
        full = ["scp", "-r", src, dst]
        print(f"    $ scp {src} -> {dst}")
        result = subprocess.run(full, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    SCP FAILED: {result.stderr.strip()}")
            return False
        return True

    # Check model exists on pod
    model_check = run_ssh("test -d /dev/shm/models/Qwen3-32B && echo OK || echo MISSING")
    if "MISSING" in (model_check.stdout or ""):
        print("    ERROR: Model not found at /dev/shm/models/Qwen3-32B on pod.")
        print("    Copy the model first: scp -r /path/to/Qwen3-32B {pod}:/dev/shm/models/")
        return

    # Create remote dirs & clean old rollouts
    run_ssh(f"rm -rf {remote_rollouts} && mkdir -p {remote_rollouts} {remote_outputs}")

    # SCP rollout JSONs to pod (create temp JSONs for text inputs)
    tmp_dir = tempfile.mkdtemp(prefix="heatmap_text_")
    try:
        for name, rollout_idx, path, stem in need_extraction:
            if rollout_idx is None:
                # Text input — create a temporary rollout JSON
                upload_path = _make_rollout_json_for_text(path, name, tmp_dir)
            else:
                upload_path = path
            if not run_scp(upload_path, f"{pod}:{remote_rollouts}/{stem}.json"):
                print(f"    ERROR: Failed to upload {stem}.json, aborting pod extraction.")
                return
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # SCP the extraction script
    script_path = os.path.join(os.path.dirname(__file__), "extract_batch_activations.py")
    if not run_scp(script_path, f"{pod}:{remote_base}/extract_batch_activations.py"):
        print("    ERROR: Failed to upload extraction script, aborting.")
        return

    # Run extraction on pod (stream output live)
    print(f"\n    Running extraction on {pod} (streaming output)...")
    run_cmd = (
        f"cd {remote_base} && "
        f"python -c '"
        f"import extract_batch_activations as m; "
        f"m.ROLLOUTS_DIR = \"{remote_rollouts}\"; "
        f"m.OUTPUT_DIR = \"{remote_outputs}\"; "
        f"m.main()'"
    )
    result = run_ssh(run_cmd, stream=True)
    if result.returncode != 0:
        print(f"    Pod extraction failed! Return code: {result.returncode}")
        return

    # SCP results back
    os.makedirs(activations_dir, exist_ok=True)
    failed = []
    for name, rollout_idx, path, stem in need_extraction:
        for suffix in ["_activations.npz", "_manifest.json"]:
            remote_file = f"{pod}:{remote_outputs}/{stem}{suffix}"
            local_file = os.path.join(activations_dir, f"{stem}{suffix}")
            if not run_scp(remote_file, local_file):
                failed.append(f"{stem}{suffix}")

    if failed:
        print(f"    WARNING: Failed to download {len(failed)} file(s): {failed}")
    else:
        print("    Pod extraction complete.")


# ── Phase 4: Generate heatmaps ───────────────────────────────────────
def _load_text(name, rollout_idx, path):
    """Load the text content from a rollout JSON or plain text file."""
    if rollout_idx is None:
        with open(path) as f:
            return f.read()
    else:
        with open(path) as f:
            return json.load(f)["chain_of_thought"]


def generate_heatmaps(items, types, output_dir, activations_dir, skip_existing):
    """Generate requested heatmap types for all items."""
    needs_semantic_sent = "sentence_semantic" in types
    needs_semantic_para = "paragraph_semantic" in types
    needs_layer44_sent = "sentence_layer44" in types
    needs_layer44_para = "paragraph_layer44" in types

    # Lazy-load sentence-transformers models
    sent_model = None
    para_model = None
    if needs_semantic_sent or needs_semantic_para:
        from sentence_transformers import SentenceTransformer
        if needs_semantic_sent:
            print("\n  Loading sentence-level semantic model...")
            sent_model = SentenceTransformer(
                "sentence-transformers/paraphrase-mpnet-base-v2", device="cpu")
        if needs_semantic_para:
            print("  Loading paragraph-level semantic model...")
            para_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", device="cpu")

    for name, rollout_idx, path in items:
        is_text = rollout_idx is None
        stem = _make_stem(name, rollout_idx)
        label = name if is_text else f"{name}/rollout_{rollout_idx}"
        print(f"\n{'='*60}")
        print(f"  {label}{'  [text]' if is_text else ''}")
        print(f"{'='*60}")

        text = _load_text(name, rollout_idx, path)

        # Split text
        sentences = split_sentences(text) if (needs_semantic_sent or needs_layer44_sent) else None
        paragraphs = split_paragraphs(text) if (needs_semantic_para or needs_layer44_para) else None

        # Load activations if needed
        npz_data = None
        manifest = None
        if needs_layer44_sent or needs_layer44_para:
            npz_path = os.path.join(activations_dir, f"{stem}_activations.npz")
            manifest_path = os.path.join(activations_dir, f"{stem}_manifest.json")
            if os.path.exists(npz_path) and os.path.exists(manifest_path):
                npz_data = np.load(npz_path)
                with open(manifest_path) as f:
                    manifest = json.load(f)
                # Use manifest's splits for layer44 (ensures token alignment)
                if needs_layer44_sent:
                    sentences = manifest["sentences"]
                if needs_layer44_para:
                    paragraphs = manifest["paragraphs"]
            else:
                print(f"    WARNING: Missing activations for {stem}, skipping layer44 types")

        # Generate each requested type
        for htype in types:
            gran, method = htype.split("_", 1)
            gran_label = gran.title()

            if gran == "sentence":
                texts = sentences
                prefix = "sentence"
            else:
                texts = paragraphs
                prefix = "paragraph"

            if texts is None or len(texts) < 2:
                print(f"    Skipping {htype} — too few {gran}s ({0 if texts is None else len(texts)})")
                continue

            # Output paths: non_cot/ for text, {prompt_name}/ for rollouts
            folder = "non_cot" if is_text else name
            out_subdir = os.path.join(output_dir, gran, method, folder)
            os.makedirs(out_subdir, exist_ok=True)

            fname = f"{stem}_{prefix}_heatmap_{method}_{'mean_' if method == 'layer44' else ''}centered"
            png_path = os.path.join(out_subdir, f"{fname}.png")
            html_path = os.path.join(out_subdir, f"{fname}.html")

            if skip_existing and os.path.exists(png_path):
                print(f"    Skipping {htype} (exists): {png_path}")
                continue

            # Compute similarity matrix
            if method == "semantic":
                model = sent_model if gran == "sentence" else para_model
                embs = model.encode(texts, convert_to_numpy=True)
                embs_centered = embs - embs.mean(axis=0)
                sim = cosine_similarity(embs_centered)
            elif method == "layer44":
                if npz_data is None:
                    print(f"    Skipping {htype} — no activation data")
                    continue
                means = npz_data[f"{prefix}_means_layer44"]
                global_mean = npz_data["global_mean_layer44"]
                centered = means - global_mean
                sim = cosine_similarity(centered)

            title = f"{label} — {gran_label} {'L44 Mean-Pooled (Centered)' if method == 'layer44' else 'Semantic (Centered)'}"
            print(f"    Plotting {htype}: {len(texts)} {gran}s")

            plot_png(sim, title, png_path, gran_label=gran_label)
            plot_html(sim, title, html_path, texts, gran_label=gran_label)


# ── CLI ──────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate pairwise similarity heatmaps for reasoning rollouts or text files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s --rollout data/reasoning_evals/rollouts/unlabeled/bagel/rollout_0.json
              %(prog)s --prompt bagel well starfish -n 2
              %(prog)s --prompt starfish -n 1 --types sentence_semantic
              %(prog)s --rollout-dir data/reasoning_evals/rollouts/unlabeled/bagel --types paragraph_semantic
              %(prog)s --text pragmatic --types sentence_semantic paragraph_semantic
        """),
    )

    # Input modes
    input_group = parser.add_argument_group("input (at least one required)")
    input_group.add_argument(
        "--rollout", nargs="+", metavar="PATH",
        help="One or more rollout JSON files",
    )
    input_group.add_argument(
        "--rollout-dir", nargs="+", metavar="PATH",
        help="One or more directories of rollout JSONs",
    )
    input_group.add_argument(
        "--prompt", nargs="+", metavar="NAME",
        help="One or more prompt names from REASONING_PROMPTS",
    )
    input_group.add_argument(
        "--text", nargs="+", metavar="NAME",
        help="One or more text names; resolved to data/texts/NAME.txt (output goes to non_cot/)",
    )

    # Options
    parser.add_argument(
        "-n", type=int, default=1,
        help="Number of rollouts per prompt (default: 1)",
    )
    parser.add_argument(
        "--types", nargs="+", choices=ALL_TYPES, default=DEFAULT_TYPES,
        help=f"Heatmap types to generate (default: {' '.join(DEFAULT_TYPES)})",
    )
    parser.add_argument(
        "--pod", default=DEFAULT_POD,
        help=f"SSH alias for GPU pod (default: {DEFAULT_POD})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--activations-dir", default=DEFAULT_ACTIVATIONS_DIR,
        help=f"Activations storage directory (default: {DEFAULT_ACTIVATIONS_DIR})",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Regenerate even if heatmaps already exist",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.rollout and not args.rollout_dir and not args.prompt and not args.text:
        parser.error("At least one of --rollout, --rollout-dir, --prompt, or --text is required")

    skip_existing = not args.no_skip_existing
    needs_layer44 = any("layer44" in t for t in args.types)

    print(f"Heatmap types: {', '.join(args.types)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Skip existing: {skip_existing}")

    # Phase 1 + 2: Resolve inputs (generates missing rollouts if --prompt)
    items = resolve_inputs(args)
    if not items:
        print("No inputs found. Nothing to do.")
        return

    print(f"\nResolved {len(items)} input(s):")
    for name, idx, path in items:
        if idx is None:
            print(f"  {name}  [text]")
        else:
            print(f"  {name}/rollout_{idx}")

    # Phase 3: Pod extraction (only if layer44 types requested)
    if needs_layer44:
        extract_activations_on_pod(items, args.activations_dir, args.pod)
    else:
        print("\n  No layer44 types requested — skipping pod extraction.")

    # Phase 4: Generate heatmaps
    generate_heatmaps(items, args.types, args.output_dir, args.activations_dir,
                      skip_existing)

    print("\nDone!")


if __name__ == "__main__":
    main()
