#!/usr/bin/env bash
# Sync data between local and s3://riya-mats-bucket
# Usage:
#   ./s3sync.sh push          # local -> S3
#   ./s3sync.sh pull          # S3 -> local
#   ./s3sync.sh push --dryrun # preview what would be uploaded
#   ./s3sync.sh pull --dryrun # preview what would be downloaded

set -euo pipefail

BUCKET="s3://riya-mats-bucket/data"
LOCAL="/home/riya/neel-projs/cot-comparisons/data"

ACTION="${1:-}"
DRYRUN=""
if [[ "${2:-}" == "--dryrun" ]]; then
    DRYRUN="--dryrun"
fi

case "$ACTION" in
    push)
        echo "Syncing local -> S3 ${DRYRUN:+(dry run)}"
        aws s3 sync "$LOCAL" "$BUCKET" $DRYRUN
        ;;
    pull)
        echo "Syncing S3 -> local ${DRYRUN:+(dry run)}"
        aws s3 sync "$BUCKET" "$LOCAL" $DRYRUN
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun]"
        exit 1
        ;;
esac
