#!/usr/bin/env bash
# Sync data between local and s3://riya-mats-bucket
# Usage:
#   ./s3sync.sh push              # local -> S3
#   ./s3sync.sh pull              # S3 -> local
#   ./s3sync.sh push --dryrun     # preview what would be uploaded
#   ./s3sync.sh pull --dryrun     # preview what would be downloaded
#   ./s3sync.sh push --a          # use Transfer Acceleration
#   ./s3sync.sh push --dryrun --a # combine flags

set -euo pipefail

BUCKET="s3://riya-mats-bucket/data/forced_response"
LOCAL="./data/forced_response"

aws configure set default.s3.max_concurrent_requests 100

ACTION="${1:-}"
shift || true

DRYRUN=""
ACCEL=""
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN="--dryrun" ;;
        --a) ACCEL="--endpoint-url https://s3-accelerate.amazonaws.com" ;;
    esac
done

case "$ACTION" in
    push)
        echo "Syncing local -> S3 ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}"
        aws s3 sync "$LOCAL" "$BUCKET" $DRYRUN $ACCEL
        ;;
    pull)
        echo "Syncing S3 -> local ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}"
        aws s3 sync "$BUCKET" "$LOCAL" $DRYRUN $ACCEL
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun] [--a]"
        exit 1
        ;;
esac
