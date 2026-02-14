#!/usr/bin/env python3
"""
Feature-vector baseline for majority/minority answer classification.

Tests whether directly labeling CoT segments with an LLM to get feature
vectors, then training a classifier, can distinguish majority from minority
rollouts — skipping all graph infrastructure.

Pipeline: chunk CoT → LLM-label each chunk → positional encoding → logistic regression.

  Phase 1 (GENERATE): Propose a feature taxonomy via GPT-4o, then score
    each 4-sentence chunk of every rollout on all features via GPT-4o-mini.

  Phase 2 (TRAIN): Load labeling results, bin chunks into positional
    segments, flatten to feature vectors, and train logistic regression
    with LOO cross-validation across all 7 prompts.

All prompts use N/M answer format from verification_rollouts.

Usage:
    # Phase 1: label chunks with LLM (~$3, ~10 min)
    python -m src2.runs.run_min_maj_feature_vector --generate

    # Phase 2: train + evaluate (~instant)
    python -m src2.runs.run_min_maj_feature_vector --train

    # Both
    python -m src2.runs.run_min_maj_feature_vector --generate --train

    # Test on a small subset
    python -m src2.runs.run_min_maj_feature_vector --generate --max-rollouts 10
"""

import argparse
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src2.tasks.forced_response.prompts import split_cot_into_sentences
from src2.tasks.min_maj_answer.task import ALL_PROMPT_IDS, MinMajAnswerTask

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "min_maj_answer"
OUTPUT_DIR = DATA_DIR / "feature_vector"
VERIFICATION_DIR = PROJECT_ROOT / "data" / "verification_rollouts"
QUESTIONS_FILE = PROJECT_ROOT / "src2" / "utils" / "questions.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TAXONOMY_MODEL = "openai/gpt-5.2"
ANNOTATION_MODEL = "openai/gpt-4o"

TASK_DESCRIPTION = (
    "Predict whether a chain-of-thought reasoning trace from a language model "
    "reaches the majority or minority answer to a question"
)
NUM_FEATURES = 10
NUM_SEGMENTS = 15
SENTENCES_PER_CHUNK = 4
MAX_WORKERS = 500

NM_SUFFIX = "_nm"
ALL_PROMPT_IDS_NM = [pid + NM_SUFFIX for pid in ALL_PROMPT_IDS]

# First 3 prompts are used for taxonomy proposal context and excluded from
# labeling/training/testing to avoid data contamination.
N_TAXONOMY_PROMPTS = 3
TAXONOMY_PROMPT_IDS_NM = ALL_PROMPT_IDS_NM[:N_TAXONOMY_PROMPTS]
EVAL_PROMPT_IDS_NM = ALL_PROMPT_IDS_NM[N_TAXONOMY_PROMPTS:]
EVAL_PROMPT_IDS = ALL_PROMPT_IDS[N_TAXONOMY_PROMPTS:]

SEED = 42


def _get_openai_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
        )
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _create_run_dir(phase: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{phase}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = OUTPUT_DIR / f"latest_{phase}"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(run_dir.name, latest)
    return run_dir


# ── Verification rollout loading ──────────────────────────────────────


def _get_latest_verification_dir(question_id: str) -> Optional[Path]:
    """Find the latest timestamped run directory for a verification question."""
    qdir = VERIFICATION_DIR / question_id
    if not qdir.exists():
        return None
    timestamped = sorted(
        [d for d in qdir.iterdir()
         if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"],
        reverse=True,
    )
    if timestamped:
        return timestamped[0]
    if (qdir / "summary.json").exists():
        return qdir
    return None


def _load_verification_rollouts(question_id: str) -> pd.DataFrame:
    """
    Load rollouts from verification_rollouts directory.

    Computes majority/minority labels from answer distribution.
    Returns a DataFrame with columns:
      prompt_id, rollout_idx, cot_content, answer, label
    """
    run_dir = _get_latest_verification_dir(question_id)
    if run_dir is None:
        raise FileNotFoundError(
            f"No verification rollouts found for {question_id} in {VERIFICATION_DIR}. "
            f"Run run_min_maj_verification.py first."
        )

    rollouts_dir = run_dir / "rollouts"
    raw_rollouts = []
    for f in sorted(rollouts_dir.iterdir()):
        if f.suffix == ".json":
            with open(f) as fh:
                raw_rollouts.append(json.load(fh))

    # Compute majority/minority labels
    answers = [r["answer"] for r in raw_rollouts if r.get("answer")]
    counts = Counter(answers)
    if not counts:
        return pd.DataFrame()
    majority_answer = counts.most_common(1)[0][0]

    rows = []
    for r in raw_rollouts:
        ans = r.get("answer", "")
        if not ans:
            continue
        rows.append({
            "prompt_id": question_id,
            "rollout_idx": r["rollout_idx"],
            "cot_content": r.get("thinking", ""),
            "answer": ans,
            "label": "majority" if ans == majority_answer else "minority",
        })

    df = pd.DataFrame(rows)
    n_maj = (df["label"] == "majority").sum()
    n_min = (df["label"] == "minority").sum()
    print(f"  {question_id}: {len(df)} rollouts, "
          f"{n_maj} majority ({majority_answer}), {n_min} minority")
    return df


# ── Chunking ──────────────────────────────────────────────────────────


def _chunk_cot(cot_text: str) -> List[str]:
    """Split CoT into 4-sentence chunks."""
    sentences = split_cot_into_sentences(cot_text)
    if not sentences:
        return []
    chunks = []
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        chunk = " ".join(sentences[i:i + SENTENCES_PER_CHUNK])
        chunks.append(chunk)
    return chunks


# ── Taxonomy proposal ─────────────────────────────────────────────────


def _load_question_prompts() -> Dict[str, str]:
    """Load question text for each _nm prompt ID."""
    from src2.utils.questions import load_custom_questions

    questions = load_custom_questions(QUESTIONS_FILE)
    prompts = {}
    for q in questions:
        if q.id in ALL_PROMPT_IDS_NM:
            prompts[q.id] = q.question
    return prompts


def _sample_examples(all_df: pd.DataFrame, n_per_class: int = 3) -> str:
    """Sample majority/minority CoT snippets from taxonomy prompts (full text, no truncation)."""
    examples = []
    for pid in TAXONOMY_PROMPT_IDS_NM:
        subset = all_df[all_df["prompt_id"] == pid]
        for label in ["majority", "minority"]:
            rows = subset[subset["label"] == label]
            if rows.empty:
                continue
            for _, row in rows.head(n_per_class).iterrows():
                cot = row["cot_content"]
                if not isinstance(cot, str) or len(cot) < 50:
                    continue
                examples.append(f"[{label.upper()} answer, prompt={pid}]\n{cot}")
    return "\n\n".join(examples)


def _propose_taxonomy(client: openai.OpenAI, all_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Use an LLM to propose a feature taxonomy grounded in actual data."""
    # Build context: question prompts (only taxonomy prompts)
    question_prompts = _load_question_prompts()
    questions_block = "\n\n".join(
        f"--- {qid} ---\n{qtext}"
        for qid, qtext in question_prompts.items()
        if qid in TAXONOMY_PROMPT_IDS_NM
    )

    # Build context: example CoTs
    examples_block = _sample_examples(all_df)

    prompt = f"""You are designing a feature taxonomy for analyzing chains of thought (reasoning traces) from language models.

The downstream task is: {TASK_DESCRIPTION}

Here are the actual questions the model is reasoning about:

{questions_block}

Here are example reasoning traces, labeled with whether they led to the majority or minority answer:

{examples_block}

Propose exactly {NUM_FEATURES} behavioral feature categories that would be relevant for distinguishing majority from minority reasoning. Each feature should describe a TYPE of reasoning behavior that might carry signal about the prediction target.

IMPORTANT: These features must be detectable from the reasoning text alone, without access to the question, the answer, or any other rollouts. The scoring model will only see a short segment of reasoning and must rate each feature based purely on what the text says.

Good features are things like: confidence level, hedging/uncertainty, backtracking, exploring multiple approaches, self-correction, verification, systematic vs intuitive reasoning, etc. Choose features specifically relevant to distinguishing majority from minority reasoning patterns.

Output a JSON array of objects with "id" (snake_case identifier) and "description" (one sentence explaining what to look for in the text).
Output ONLY the JSON array, no other text.

Example format:
[
  {{"id": "confident_commitment", "description": "The text shows the model confidently committing to a specific conclusion without reservation"}},
  {{"id": "hedging", "description": "The text shows the model expressing uncertainty or actively considering multiple options"}}
]"""

    print(f"Proposing feature taxonomy with {TAXONOMY_MODEL} (reasoning=high)...")
    response = client.chat.completions.create(
        model=TAXONOMY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
        temperature=0.7,
        extra_body={"reasoning": {"effort": "high"}},
    )
    raw = response.choices[0].message.content or ""

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        features = json.loads(raw[start:end])
    else:
        features = json.loads(raw)

    print(f"Proposed {len(features)} features:")
    for f in features:
        print(f"  - {f['id']}: {f['description']}")
    return features


# ── Chunk scoring ─────────────────────────────────────────────────────


def _score_chunk(
    client: openai.OpenAI,
    chunk_text: str,
    taxonomy: List[Dict[str, str]],
) -> Dict[str, float]:
    """Score a single chunk on all features in the taxonomy."""
    feature_block = "\n".join(
        f'  - "{f["id"]}": {f["description"]}' for f in taxonomy
    )
    feature_ids = [f["id"] for f in taxonomy]

    prompt = f"""You are analyzing a segment of reasoning from a language model's chain of thought.

Task context: {TASK_DESCRIPTION}

Reasoning segment:
\"{chunk_text}\"

Feature taxonomy:
{feature_block}

For each feature, rate how strongly this segment exhibits that feature on a scale from 0.0 to 1.0.
Output a JSON object mapping feature_id to score. Output ONLY the JSON object, no other text.

Example: {{"problem_setup": 0.1, "calculation": 0.9, ...}}"""

    try:
        response = client.chat.completions.create(
            model=ANNOTATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""

        # Parse JSON — find first { and last }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            scores = json.loads(raw[start:end])
        else:
            scores = json.loads(raw)

        # Ensure all features present, clamp to [0, 1]
        clean_scores = {}
        for fid in feature_ids:
            val = scores.get(fid, 0.0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            clean_scores[fid] = max(0.0, min(1.0, val))
        return clean_scores

    except Exception as e:
        # Fallback: all zeros
        return {fid: 0.0 for fid in feature_ids}


# ── Phase 1: Generate labeling data ──────────────────────────────────


def _save_labeling(results: List[Dict], save_path: Path, taxonomy: List[Dict[str, str]]):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "task_description": TASK_DESCRIPTION,
                "taxonomy_model": TAXONOMY_MODEL,
                "annotation_model": ANNOTATION_MODEL,
                "taxonomy_prompt_ids": TAXONOMY_PROMPT_IDS_NM,
                "eval_prompt_ids": EVAL_PROMPT_IDS_NM,
                "n_rollouts": len(results),
                "taxonomy": taxonomy,
                "rollouts": results,
            },
            f,
            indent=2,
        )


def generate_labeling_data(max_rollouts: Optional[int] = None):
    """Score all CoT chunks on feature taxonomy via LLM."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_openai_client()

    # ── Load all rollouts from verification_rollouts ──────────────────
    # Load ALL prompts (taxonomy prompts needed for taxonomy proposal context)
    all_dfs = []
    for qid in ALL_PROMPT_IDS_NM:
        df = _load_verification_rollouts(qid)
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rollouts (all prompts): {len(all_df)} ({all_df['label'].value_counts().to_dict()})")

    print(f"  Taxonomy prompts (used for taxonomy context): {TAXONOMY_PROMPT_IDS_NM}")
    print(f"  All prompts will be labeled and used for LOO-CV")

    # ── Propose taxonomy (or load existing) ───────────────────────────
    taxonomy_path = OUTPUT_DIR / "taxonomy.json"
    if taxonomy_path.exists():
        print(f"Loading existing taxonomy from {taxonomy_path}")
        with open(taxonomy_path) as f:
            taxonomy_data = json.load(f)
        taxonomy = taxonomy_data["features"]
    else:
        taxonomy = _propose_taxonomy(client, all_df)
        taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(taxonomy_path, "w") as f:
            json.dump(
                {
                    "task_description": TASK_DESCRIPTION,
                    "features": taxonomy,
                    "taxonomy_prompt_ids": TAXONOMY_PROMPT_IDS_NM,
                    "eval_prompt_ids": EVAL_PROMPT_IDS_NM,
                },
                f,
                indent=2,
            )
        print(f"Saved taxonomy to {taxonomy_path}")

    feature_ids = [feat["id"] for feat in taxonomy]

    # ── Build rollout list (all prompts) ─────────────────────────────
    all_rollouts = []
    for _, row in all_df.iterrows():
        cot = row["cot_content"]
        if not isinstance(cot, str) or len(cot) < 50:
            continue
        all_rollouts.append({
            "prompt_id": row["prompt_id"],
            "rollout_idx": int(row["rollout_idx"]),
            "answer": str(row["answer"]),
            "label": row["label"],
            "cot_content": cot,
        })

    if max_rollouts is not None:
        all_rollouts = all_rollouts[:max_rollouts]
        print(f"Limited to {max_rollouts} rollouts for testing")

    # ── Resume from checkpoint ────────────────────────────────────────
    checkpoint_path = OUTPUT_DIR / "labeling_checkpoint.json"
    done_keys = set()
    prior_results = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            prior_data = json.load(f)
        prior_results = prior_data.get("rollouts", [])
        for r in prior_results:
            done_keys.add((r["prompt_id"], r["rollout_idx"]))
        print(f"Resuming: {len(prior_results)} rollouts already done")

    remaining = [
        r for r in all_rollouts
        if (r["prompt_id"], r["rollout_idx"]) not in done_keys
    ]
    print(f"Remaining: {len(remaining)} rollouts to label")

    if not remaining:
        print("All rollouts already complete!")
        return

    # ── Build all (rollout, chunk) work items ─────────────────────────
    work_items = []  # (rollout_info, chunk_idx, chunk_text)
    for rollout in remaining:
        chunks = _chunk_cot(rollout["cot_content"])
        for ci, chunk_text in enumerate(chunks):
            work_items.append((rollout, ci, chunk_text))

    print(f"Total LLM calls: {len(work_items)} chunks across {len(remaining)} rollouts")

    # ── Score all chunks in parallel ──────────────────────────────────
    # Collect results keyed by (prompt_id, rollout_idx)
    chunk_results: Dict[tuple, Dict] = {}
    results_lock = threading.Lock()
    n_errors = 0

    def _score_work_item(item):
        rollout_info, chunk_idx, chunk_text = item
        scores = _score_chunk(client, chunk_text, taxonomy)
        return rollout_info, chunk_idx, scores

    n_workers = min(MAX_WORKERS, len(work_items))
    print(f"Scoring {len(work_items)} chunks with {n_workers} parallel workers")

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_score_work_item, item): item
                for item in work_items
            }
            with tqdm(total=len(work_items), desc="Scoring chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        rollout_info, chunk_idx, scores = future.result()
                    except Exception as e:
                        n_errors += 1
                        if n_errors <= 5:
                            print(f"\n  Error: {e}")
                        elif n_errors == 6:
                            print("\n  (suppressing further error messages)")
                        pbar.update(1)
                        continue

                    key = (rollout_info["prompt_id"], rollout_info["rollout_idx"])
                    with results_lock:
                        if key not in chunk_results:
                            chunks = _chunk_cot(rollout_info["cot_content"])
                            chunk_results[key] = {
                                "prompt_id": rollout_info["prompt_id"],
                                "rollout_idx": rollout_info["rollout_idx"],
                                "answer": rollout_info["answer"],
                                "label": rollout_info["label"],
                                "n_chunks": len(chunks),
                                "chunk_scores": [None] * len(chunks),
                            }
                        chunk_results[key]["chunk_scores"][chunk_idx] = scores
                        pbar.update(1)

                        # Checkpoint every 50 completed rollouts
                        n_complete = sum(
                            1 for cr in chunk_results.values()
                            if all(cs is not None for cs in cr["chunk_scores"])
                        )
                        if n_complete > 0 and n_complete % 50 == 0:
                            _save_checkpoint(
                                prior_results, chunk_results,
                                feature_ids, taxonomy, checkpoint_path,
                            )
    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving checkpoint...")
    finally:
        _save_checkpoint(
            prior_results, chunk_results,
            feature_ids, taxonomy, checkpoint_path,
        )
        n_complete = sum(
            1 for cr in chunk_results.values()
            if all(cs is not None for cs in cr["chunk_scores"])
        )
        print(f"Checkpoint saved: {len(prior_results) + n_complete} total rollouts "
              f"({n_errors} errors)")

    if n_errors > 0:
        print(f"Warning: {n_errors} chunk scoring calls failed. Rerun --generate to retry.")

    # ── Save final results ────────────────────────────────────────────
    new_results = _finalize_chunk_results(chunk_results, feature_ids)
    all_results = prior_results + new_results
    run_dir = _create_run_dir("generate")
    final_path = run_dir / "labeling_results.json"
    _save_labeling(all_results, final_path, taxonomy)
    print(f"\nLabeling complete. Saved {len(all_results)} rollouts to {final_path}")


def _finalize_chunk_results(
    chunk_results: Dict[tuple, Dict],
    feature_ids: List[str],
) -> List[Dict]:
    """Convert chunk_results dict to list, filling in missing scores with zeros.

    Creates copies to avoid mutating chunk_results in-place (which would
    inflate the n_complete counter used for checkpoint frequency).
    """
    results = []
    for key, cr in chunk_results.items():
        entry = {k: v for k, v in cr.items() if k != "chunk_scores"}
        entry["chunk_scores"] = [
            cs if cs is not None else {fid: 0.0 for fid in feature_ids}
            for cs in cr["chunk_scores"]
        ]
        results.append(entry)
    return results


def _save_checkpoint(
    prior_results: List[Dict],
    chunk_results: Dict[tuple, Dict],
    feature_ids: List[str],
    taxonomy: List[Dict[str, str]],
    checkpoint_path: Path,
):
    """Save completed rollouts to checkpoint.

    Only saves rollouts where ALL chunks have been scored, so that
    incomplete rollouts are retried on resume rather than permanently
    stored with all-zero fallbacks.
    """
    complete = [
        cr for cr in chunk_results.values()
        if all(cs is not None for cs in cr["chunk_scores"])
    ]
    all_results = prior_results + complete
    _save_labeling(all_results, checkpoint_path, taxonomy)


# ── Phase 2: Train logistic regression baseline ──────────────────────


def _extract_positional_features(
    chunk_scores: List[Dict[str, float]],
    feature_ids: List[str],
) -> np.ndarray:
    """Extract positionally-encoded feature vector from chunk scores.

    Bins chunks into NUM_SEGMENTS segments (early/mid/late), sums feature
    scores per segment, and flattens to a 1D vector.

    Returns array of shape [NUM_SEGMENTS * len(feature_ids)].
    """
    n_features = len(feature_ids)
    feature_matrix = np.zeros((NUM_SEGMENTS, n_features))
    segment_counts = np.zeros(NUM_SEGMENTS)

    if not chunk_scores:
        return feature_matrix.flatten()

    n_chunks = len(chunk_scores)
    for i, scores in enumerate(chunk_scores):
        position = i / max(n_chunks - 1, 1)
        seg = min(int(position * NUM_SEGMENTS), NUM_SEGMENTS - 1)
        segment_counts[seg] += 1
        for j, fid in enumerate(feature_ids):
            feature_matrix[seg, j] += scores.get(fid, 0.0)

    for s in range(NUM_SEGMENTS):
        if segment_counts[s] > 0:
            feature_matrix[s, :] /= segment_counts[s]

    return feature_matrix.flatten()


def train_baseline():
    """Train multiple classifiers on feature vectors using LOO-CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # Find labeling results
    latest_generate = OUTPUT_DIR / "latest_generate"
    if latest_generate.exists():
        results_path = latest_generate / "labeling_results.json"
    else:
        results_path = OUTPUT_DIR / "labeling_checkpoint.json"
    if not results_path.exists():
        print(f"No labeling results found in {OUTPUT_DIR}")
        print("Run with --generate first.")
        return
    print(f"Loading labeling results from {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    rollouts = data["rollouts"]
    taxonomy = data["taxonomy"]
    feature_ids = [feat["id"] for feat in taxonomy]
    print(f"Loaded {len(rollouts)} rollouts, {len(feature_ids)} features")

    # ── Extract features and labels ───────────────────────────────────
    X_list = []
    y_list = []
    prompt_ids = []
    metadata = []

    for rollout in rollouts:
        if not rollout.get("chunk_scores"):
            continue
        features = _extract_positional_features(rollout["chunk_scores"], feature_ids)
        label = 1 if rollout["label"] == "minority" else 0
        X_list.append(features)
        y_list.append(label)
        prompt_ids.append(rollout["prompt_id"])
        metadata.append({
            "prompt_id": rollout["prompt_id"],
            "rollout_idx": rollout["rollout_idx"],
            "answer": rollout["answer"],
            "label": rollout["label"],
        })

    X = np.stack(X_list)
    y = np.array(y_list)
    prompt_ids_arr = np.array(prompt_ids)
    print(f"Total samples: {len(y)}")
    print(f"  majority (0): {(y == 0).sum()}, minority (1): {(y == 1).sum()}")

    feature_dim = NUM_SEGMENTS * len(feature_ids)
    feature_names = [
        f"seg{s}_{fid}" for s in range(NUM_SEGMENTS) for fid in feature_ids
    ]
    print(f"Feature dimensionality: {feature_dim}")

    # Per-class metrics helper
    def _metrics(y_true, y_pred, pos_label=1):
        tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
        fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
        fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # ── Define classifiers to compare ─────────────────────────────────
    minority_ratio = float((y == 1).sum()) / float((y == 0).sum()) if (y == 0).sum() > 0 else 1.0

    classifiers = {
        "LogReg (balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED,
        ),
        "LogReg (L1, C=0.1)": LogisticRegression(
            class_weight="balanced", penalty="l1", solver="liblinear",
            C=0.1, max_iter=1000, random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED,
        ),
    }

    folds = MinMajAnswerTask.loo_folds()
    best_clf_name = None
    best_macro_f1 = -1.0
    all_classifier_results = {}

    for clf_name, clf_template in classifiers.items():
        fold_results = []
        all_test_preds = []
        all_test_true = []

        print(f"\n{'=' * 60}")
        print(f"  {clf_name} — LOO Cross-Validation ({len(folds)} folds)")
        print(f"{'=' * 60}")

        for fold in folds:
            test_pid_nm = fold["test_prompt_id"] + NM_SUFFIX
            train_pids_nm = {pid + NM_SUFFIX for pid in fold["train_prompt_ids"]}

            train_mask = np.isin(prompt_ids_arr, list(train_pids_nm))
            test_mask = prompt_ids_arr == test_pid_nm

            if not test_mask.any():
                print(f"  Fold {fold['fold_idx']} ({test_pid_nm}): no test data, skipping")
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Standardize per fold
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Clone the classifier for this fold
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(X_train_s, y_train)

            test_pred = clf.predict(X_test_s)
            test_acc = float((y_test == test_pred).mean())
            min_m = _metrics(y_test, test_pred, pos_label=1)
            maj_m = _metrics(y_test, test_pred, pos_label=0)
            macro_f1 = (min_m["f1"] + maj_m["f1"]) / 2

            print(f"  Fold {fold['fold_idx']} ({fold['test_prompt_id']}): "
                  f"Acc={test_acc:.3f}, F1={macro_f1:.3f} "
                  f"(maj={maj_m['f1']:.3f}, min={min_m['f1']:.3f}), "
                  f"n={len(y_test)}")

            fold_results.append({
                "fold_idx": fold["fold_idx"],
                "test_prompt_id": fold["test_prompt_id"],
                "accuracy": test_acc,
                "macro_f1": macro_f1,
                "majority_f1": maj_m["f1"],
                "majority_precision": maj_m["precision"],
                "majority_recall": maj_m["recall"],
                "minority_f1": min_m["f1"],
                "minority_precision": min_m["precision"],
                "minority_recall": min_m["recall"],
                "n_test": len(y_test),
                "n_train": len(y_train),
            })
            all_test_preds.extend(test_pred.tolist())
            all_test_true.extend(y_test.tolist())

        # ── Aggregate metrics for this classifier ─────────────────────
        all_test_preds_arr = np.array(all_test_preds)
        all_test_true_arr = np.array(all_test_true)
        pooled_acc = float((all_test_true_arr == all_test_preds_arr).mean())
        pooled_min = _metrics(all_test_true_arr, all_test_preds_arr, pos_label=1)
        pooled_maj = _metrics(all_test_true_arr, all_test_preds_arr, pos_label=0)
        pooled_macro_f1 = (pooled_min["f1"] + pooled_maj["f1"]) / 2
        fold_f1s = [fr["macro_f1"] for fr in fold_results]
        mean_fold_f1 = sum(fold_f1s) / len(fold_f1s) if fold_f1s else 0.0

        majority_frac = float((all_test_true_arr == 0).sum()) / len(all_test_true_arr) if len(all_test_true_arr) > 0 else 0.0

        print(f"\n  Aggregate ({len(fold_results)} folds):")
        print(f"    Pooled accuracy:    {pooled_acc:.3f} (n={len(all_test_true_arr)})")
        print(f"    Pooled macro F1:    {pooled_macro_f1:.3f}")
        print(f"    Mean fold macro F1: {mean_fold_f1:.3f}")
        print(f"    Pooled majority F1: {pooled_maj['f1']:.3f}")
        print(f"    Pooled minority F1: {pooled_min['f1']:.3f}")
        print(f"    Chance baseline (always majority): {majority_frac:.3f}")

        all_classifier_results[clf_name] = {
            "pooled_accuracy": pooled_acc,
            "pooled_macro_f1": pooled_macro_f1,
            "mean_fold_macro_f1": mean_fold_f1,
            "pooled_majority_f1": pooled_maj["f1"],
            "pooled_majority_precision": pooled_maj["precision"],
            "pooled_majority_recall": pooled_maj["recall"],
            "pooled_minority_f1": pooled_min["f1"],
            "pooled_minority_precision": pooled_min["precision"],
            "pooled_minority_recall": pooled_min["recall"],
            "chance_baseline": majority_frac,
            "n_folds": len(fold_results),
            "n_total": len(all_test_true_arr),
            "fold_results": fold_results,
        }

        if pooled_macro_f1 > best_macro_f1:
            best_macro_f1 = pooled_macro_f1
            best_clf_name = clf_name

    # ── Summary comparison ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Classifier':<25} {'Accuracy':>8} {'Macro F1':>9} {'Maj F1':>7} {'Min F1':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*7} {'-'*7}")
    for clf_name, res in all_classifier_results.items():
        marker = " *" if clf_name == best_clf_name else ""
        print(f"  {clf_name:<25} {res['pooled_accuracy']:>8.3f} {res['pooled_macro_f1']:>9.3f} "
              f"{res['pooled_majority_f1']:>7.3f} {res['pooled_minority_f1']:>7.3f}{marker}")
    print(f"\n  * Best by pooled macro F1")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "method": "loo_cv",
        "baseline": "feature_vector",
        "best_classifier": best_clf_name,
        "best_pooled_macro_f1": best_macro_f1,
        "feature_names": feature_names,
        "num_segments": NUM_SEGMENTS,
        "num_features": len(feature_ids),
        "taxonomy": taxonomy,
        "classifiers": all_classifier_results,
    }
    run_dir = _create_run_dir("train")
    with open(run_dir / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {run_dir / 'baseline_results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Feature-vector baseline for majority/minority classification",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Phase 1: label CoT chunks with LLM features",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Phase 2: train logistic regression on feature vectors",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="Limit number of rollouts (for testing)",
    )
    args = parser.parse_args()

    if not args.generate and not args.train:
        parser.print_help()
        print("\nSpecify --generate, --train, or both.")
        return

    if args.generate:
        generate_labeling_data(max_rollouts=args.max_rollouts)

    if args.train:
        train_baseline()


if __name__ == "__main__":
    main()
