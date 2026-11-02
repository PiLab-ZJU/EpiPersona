#!/usr/bin/env bash
# Shared environment for the scripts in this directory.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export EPIPERSONA_DIST_TIMEOUT_SEC="${EPIPERSONA_DIST_TIMEOUT_SEC:-10800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"

if [[ -f "${ROOT}/.env" ]]; then
  while IFS= read -r _line || [[ -n "${_line}" ]]; do
    _line="${_line#"${_line%%[![:space:]]*}"}"
    [[ -z "${_line}" || "${_line}" == \#* || "${_line}" != *=* ]] && continue
    _key="${_line%%=*}"
    _value="${_line#*=}"
    _key="${_key%"${_key##*[![:space:]]}"}"
    _value="${_value#"${_value%%[![:space:]]*}"}"
    _value="${_value%\"}"
    _value="${_value#\"}"
    _value="${_value%\'}"
    _value="${_value#\'}"
    if [[ -n "${_key}" && -z "${!_key+x}" ]]; then
      export "${_key}=${_value}"
    fi
  done < "${ROOT}/.env"
fi

cd "${ROOT}"
