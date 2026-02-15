#!/usr/bin/env bash
# Sync only .npz activation files under data/forced_response/ from S3.
#
# Usage:
#   ./s3sync_forced_npz.sh pull              # S3 -> local (.npz only)
#   ./s3sync_forced_npz.sh push              # local -> S3 (.npz only)
#   ./s3sync_forced_npz.sh pull --dryrun     # preview what would be downloaded
#   ./s3sync_forced_npz.sh pull --a          # use Transfer Acceleration

set -euo pipefail

BUCKET="s3://riya-mats-bucket/data/min_maj_answer/"
LOCAL="./data/min_maj_answer/"

aws configure set default.s3.max_concurrent_requests 100

ACTION="${1:-}"
shift || true

DRYRUN=""
ACCEL=""
EXCLUDE_NPZ=""
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN="--dryrun" ;;
        --a) ACCEL="--endpoint-url https://s3-accelerate.amazonaws.com" ;;
        --no-npz) EXCLUDE_NPZ="--exclude *.npz" ;;
    esac
done

case "$ACTION" in
    push)
        echo "Syncing: local -> S3 ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}${EXCLUDE_NPZ:+ (no .npz)}"
        aws s3 sync "$LOCAL" "$BUCKET" \
            $DRYRUN $ACCEL $EXCLUDE_NPZ
        ;;
    pull)
        echo "Syncing: S3 -> local ${DRYRUN:+(dry run)}${ACCEL:+ (accelerated)}${EXCLUDE_NPZ:+ (no .npz)}"
        aws s3 sync "$BUCKET" "$LOCAL" \
            $DRYRUN $ACCEL $EXCLUDE_NPZ
        ;;
    *)
        echo "Usage: $0 {push|pull} [--dryrun] [--a] [--no-npz]"
        exit 1
        ;;
esac
