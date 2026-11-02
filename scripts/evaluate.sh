#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
PREDICTIONS="${1:?usage: scripts/evaluate.sh /path/to/inference_predictions.json}"
python -m epipersona.evaluate --predictions "${PREDICTIONS}"
