#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <user@remote-host> <config-path> [remote-root]"
  exit 1
fi

REMOTE_HOST="$1"
CONFIG_PATH="$2"
REMOTE_ROOT="${3:-~/trainclaw}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep transfer small and deterministic.
rsync -az --delete \
  --exclude '.git' \
  --exclude 'results' \
  --exclude '__pycache__' \
  "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

ssh "${REMOTE_HOST}" "cd ${REMOTE_ROOT} && chmod +x scripts/run_local.sh && ./scripts/run_local.sh ${CONFIG_PATH}"
