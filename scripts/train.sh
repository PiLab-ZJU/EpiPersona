#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
CONFIG="${1:?usage: scripts/train.sh configs/llama3b.yaml}"

mapfile -t _CFG < <(python - "${CONFIG}" <<'PY'
import sys
from epipersona.config import load_config
config = load_config(sys.argv[1])
print(config.master_port)
print(config.num_gpus)
PY
)
PORT="${_CFG[0]}"
EXPECT_GPUS="${_CFG[1]}"
GPU_IDS="${EPIPERSONA_GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a _IDS <<< "${GPU_IDS}"
if [[ "${#_IDS[@]}" -ne "${EXPECT_GPUS}" ]]; then
  echo "EPIPERSONA_GPU_IDS has ${#_IDS[@]} devices; this config was run with ${EXPECT_GPUS}." >&2
  echo "A different device count changes the global batch." >&2
  exit 1
fi
export EPIPERSONA_DIST_SYNC_ID="${PORT}_$$"
LOG_DIR="$(python - <<'PY'
from epipersona.paths import log_root
print(log_root())
PY
)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(basename "${CONFIG}" .yaml)_$(date +%Y%m%d%H%M%S).log"
echo "log_file ${LOG_FILE}"

deepspeed --include "localhost:${GPU_IDS}" --master_port "${PORT}" \
  epipersona/train.py --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
