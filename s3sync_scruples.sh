#!/usr/bin/env bash
# Sync only the scruples data needed for the uncertainty-robust split from S3.
# Runs Python to compute the exact anecdote IDs in the split, then syncs
# only those anecdote-level folders (runs/<timestamp>/<anecdote_id>/).
#
# Usage:
#   ./s3sync_scruples.sh push              # local -> S3
#   ./s3sync_scruples.sh pull              # S3 -> local
#   ./s3sync_scruples.sh push --dryrun     # preview
#   ./s3sync_scruples.sh pull --no-npz     # skip .npz activation files

set -euo pipefail

BUCKET="s3://riya-mats-bucket/data/scruples"
LOCAL="./data/scruples"

# ── Only these variants are needed for the uncertainty-robust split ──
VARIANTS=(
    suggest_wrong
    suggest_right
)

aws configure set default.s3.max_concurrent_requests 100

ACTION="${1:-}"
shift || true

DRYRUN=""
ACCEL=""
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN="--dryrun" ;;
        --a) ACCEL="--endpoint-url https://s3-accelerate.amazonaws.com" ;;
        --no-npz) EXTRA_ARGS+=("--exclude" "*.npz") ;;
    esac
done

sync_cmd() {
    local src="$1" dst="$2"
    aws s3 sync "$src" "$dst" $DRYRUN $ACCEL "${EXTRA_ARGS[@]}"
}

case "$ACTION" in
    push|pull)
        if [ "$ACTION" = "push" ]; then
            SRC_BASE="$LOCAL"
            DST_BASE="$BUCKET"
        else
            SRC_BASE="$BUCKET"
            DST_BASE="$LOCAL"
        fi

        echo "=== Scruples sync: $ACTION ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)} ==="

        # 1. Sync prompt/results CSVs for each variant
        echo "[1/3] Syncing CSVs..."
        for variant in "${VARIANTS[@]}"; do
            for prefix in prompts results; do
                echo "  ${prefix}_${variant}.csv"
                aws s3 cp \
                    "${SRC_BASE}/${prefix}_${variant}.csv" \
                    "${DST_BASE}/${prefix}_${variant}.csv" \
                    $DRYRUN $ACCEL 2>&1 | grep -v "^$" || true
            done
        done

        # 2. Run Python to get the exact anecdote-level run paths needed
        echo ""
        echo "[2/3] Computing uncertainty-robust split to find needed run paths..."
        VARIANTS_JSON=$(printf '%s\n' "${VARIANTS[@]}" | jq -R . | jq -s .)
        RUN_PATHS=$(python3 -c "
import pandas as pd
from src2.tasks import ScruplesTask
from pathlib import Path

variants = $VARIANTS_JSON
task = ScruplesTask(
    subject_model='moonshotai/kimi-k2-thinking',
    variant=variants[0],
    data_dir=Path('$LOCAL'),
)
split = task.get_uncertainty_robust_split(
    switch_threshold=0.40, non_syc_max_switch=0.10,
    high_intervention_rate=0.82, low_intervention_rate=0.70,
    n_syc_high_per_variant=25, n_syc_low_per_variant=25,
    n_non_syc_per_variant=50,
    variants=variants,
)
all_ids = set(split['syc_ids']) | set(split['non_syc_ids'])

# Find the run_path directories for these anecdotes
paths = set()
for variant in variants:
    df = pd.read_csv(f'$LOCAL/results_{variant}.csv')
    df = df[df['anecdote_id'].isin(all_ids)]
    for rp in df['run_path']:
        # run_path = 'runs/<timestamp>/<anecdote_id>/control_0.json'
        # we want 'runs/<timestamp>/<anecdote_id>'
        parts = rp.split('/')
        paths.add('/'.join(parts[:3]))

for p in sorted(paths):
    print(p)
" 2>&1 | grep '^runs/')

        N_PATHS=$(echo "$RUN_PATHS" | wc -l)
        echo "  Found $N_PATHS anecdote folders to sync"

        # 3. Sync only those anecdote-level folders
        echo ""
        echo "[3/3] Syncing anecdote folders..."
        while IFS= read -r rel_path; do
            echo "  $rel_path"
            sync_cmd "$SRC_BASE/$rel_path" "$DST_BASE/$rel_path"
        done <<< "$RUN_PATHS"

        echo "Done."
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun] [--a] [--no-npz]"
        exit 1
        ;;
esac
