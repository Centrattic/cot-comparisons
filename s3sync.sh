#!/usr/bin/env bash
# Sync data between local and s3://riya-mats-bucket
# Usage:
#   ./s3sync.sh push              # local -> S3
#   ./s3sync.sh pull              # S3 -> local
#   ./s3sync.sh push --dryrun     # preview what would be uploaded
#   ./s3sync.sh pull --dryrun     # preview what would be downloaded
#   ./s3sync.sh push --a          # use Transfer Acceleration
#   ./s3sync.sh push --no-npz     # exclude .npz files
#   ./s3sync.sh push --dryrun --a # combine flags

set -euo pipefail

BUCKET="s3://riya-mats-bucket/data/"
LOCAL="./data/"

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

case "$ACTION" in
    push)
        echo "Syncing local -> S3 ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}${EXCLUDE:+ (no .npz)}"
        aws s3 sync "$LOCAL" "$BUCKET" $DRYRUN $ACCEL $EXCLUDE
        ;;
    pull)
        echo "Syncing S3 -> local ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}${EXCLUDE:+ (no .npz)}"
        aws s3 sync "$BUCKET" "$LOCAL" $DRYRUN $ACCEL $EXCLUDE
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun] [--a] [--no-npz]"
        exit 1
        ;;
esac
