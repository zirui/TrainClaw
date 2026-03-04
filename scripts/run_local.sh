#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/exp/phase1_smoke.yaml}"

cd "${ROOT_DIR}"
python3 runners/launcher.py --config "${CONFIG_PATH}"
