#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${PROJECT_DIR}/.tools/uv-0.12.5/uv"

if [[ ! -x "${UV_BIN}" ]]; then
  printf 'uv is not bootstrapped; run scripts/bootstrap_uv.sh first.\n' >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/xhand1-uv-cache}"
export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
unset PYTHONHOME
unset PYTHONPATH

cd "${PROJECT_DIR}"
exec "${UV_BIN}" "$@"
