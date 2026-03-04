#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-path> [job-name]"
  exit 1
fi

CONFIG_PATH="$1"
JOB_NAME="${2:-trainclaw-phase1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sbatch \
  --job-name="${JOB_NAME}" \
  --output="${ROOT_DIR}/results/slurm-%j.out" \
  --chdir="${ROOT_DIR}" \
  --wrap="./scripts/run_local.sh ${CONFIG_PATH}"
