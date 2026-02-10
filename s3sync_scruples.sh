#!/usr/bin/env bash
# Sync only the scruples data we actually need from s3://riya-mats-bucket.
# Syncs: CSVs, prompts, and only the 12 run folders referenced by results CSVs.
#
# Usage:
#   ./s3sync_scruples.sh push              # local -> S3
#   ./s3sync_scruples.sh pull              # S3 -> local
#   ./s3sync_scruples.sh push --dryrun     # preview
#   ./s3sync_scruples.sh pull --no-npz     # skip .npz activation files

set -euo pipefail

BUCKET="s3://riya-mats-bucket/data/scruples"
LOCAL="./data/scruples"

aws configure set default.s3.max_concurrent_requests 100

ACTION="${1:-}"
shift || true

DRYRUN=""
ACCEL=""
EXCLUDE=""
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN="--dryrun" ;;
        --a) ACCEL="--endpoint-url https://s3-accelerate.amazonaws.com" ;;
        --no-npz) EXCLUDE="--exclude *.npz" ;;
    esac
done

# Run folders referenced by results CSVs (suggest_wrong, suggest_right, first_person)
NEEDED_RUNS=(
    # suggest_wrong
    "2026-02-02_02-59-08"
    "2026-02-02_06-46-59"
    "2026-02-02_06-48-55"
    "2026-02-02_07-19-56"
    "2026-02-08_05-02-53"
    "2026-02-08_06-03-49"
    # suggest_right
    "2026-02-02_22-54-16"
    "2026-02-08_05-02-55"
    "2026-02-08_06-03-51"
    # first_person
    "2026-02-02_02-56-31"
    "2026-02-02_06-56-46"
    "2026-02-02_06-57-08"
)

sync_cmd() {
    local src="$1" dst="$2"
    aws s3 sync "$src" "$dst" $DRYRUN $ACCEL $EXCLUDE
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

        echo "=== Scruples sync: $ACTION ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}${EXCLUDE:+ (no .npz)} ==="

        # Sync top-level files (CSVs, prompts, etc.) — exclude the runs/ subdirectory
        echo "[1/2] Syncing top-level files..."
        aws s3 sync "$SRC_BASE" "$DST_BASE" \
            --exclude "runs/*" \
            $DRYRUN $ACCEL $EXCLUDE

        # Sync only the needed run folders
        echo "[2/2] Syncing ${#NEEDED_RUNS[@]} run folders..."
        for folder in "${NEEDED_RUNS[@]}"; do
            echo "  runs/$folder"
            sync_cmd "$SRC_BASE/runs/$folder" "$DST_BASE/runs/$folder"
        done

        echo "Done."
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun] [--a] [--no-npz]"
        exit 1
        ;;
esac
