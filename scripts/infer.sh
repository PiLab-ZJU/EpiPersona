#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
CHECKPOINT="${1:?usage: scripts/infer.sh /path/to/checkpoint-N [batch_size]}"
BATCH="${2:-}"
if [[ -n "${BATCH}" ]]; then
  python -m epipersona.infer --checkpoint "${CHECKPOINT}" --batch-size "${BATCH}"
else
  python -m epipersona.infer --checkpoint "${CHECKPOINT}"
fi
