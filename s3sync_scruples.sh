#!/usr/bin/env bash
# Sync scruples data with s3://riya-mats-bucket.
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
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN="--dryrun" ;;
        --a) ACCEL="--endpoint-url https://s3-accelerate.amazonaws.com" ;;
        --no-npz) EXTRA_ARGS+=("--exclude" "*.npz") ;;
    esac
done

case "$ACTION" in
    push|pull)
        if [ "$ACTION" = "push" ]; then
            SRC="$LOCAL"
            DST="$BUCKET"
        else
            SRC="$BUCKET"
            DST="$LOCAL"
        fi

        echo "=== Scruples sync: $ACTION ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)} ==="
        aws s3 sync "$SRC" "$DST" $DRYRUN $ACCEL "${EXTRA_ARGS[@]}"
        echo "Done."
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun] [--a] [--no-npz]"
        exit 1
        ;;
esac
