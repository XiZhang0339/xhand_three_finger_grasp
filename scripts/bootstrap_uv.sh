#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_VERSION="0.12.5"
UV_ARCHIVE="uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256="68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2"
UV_BINARY_SHA256="b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_ARCHIVE}"
UV_INSTALL_DIR="${PROJECT_DIR}/.tools/uv-${UV_VERSION}"
UV_BIN="${UV_INSTALL_DIR}/uv"

if [[ ! -x "${UV_BIN}" ]]; then
  mkdir -p "${PROJECT_DIR}/.tools"
  bootstrap_prefix="${PROJECT_DIR}/.tools/.uv-bootstrap."
  bootstrap_dir="$(mktemp -d "${bootstrap_prefix}XXXXXXXX")"
  cleanup() {
    if [[ -n "${bootstrap_dir:-}" && "${bootstrap_dir}" == "${bootstrap_prefix}"* ]]; then
      rm -rf -- "${bootstrap_dir}"
    fi
  }
  trap cleanup EXIT
  download_path="${bootstrap_dir}/${UV_ARCHIVE}"
  staged_install="${bootstrap_dir}/uv-${UV_VERSION}"
  curl --proto '=https' --tlsv1.2 -fL "${UV_URL}" -o "${download_path}"
  printf '%s  %s\n' "${UV_SHA256}" "${download_path}" | sha256sum --check --status
  mkdir -p "${staged_install}"
  tar -xzf "${download_path}" -C "${staged_install}" --strip-components=1
  printf '%s  %s\n' "${UV_BINARY_SHA256}" "${staged_install}/uv" \
    | sha256sum --check --status
  if [[ -e "${UV_INSTALL_DIR}" ]]; then
    printf 'Refusing to overwrite incomplete uv installation: %s\n' "${UV_INSTALL_DIR}" >&2
    exit 1
  fi
  mv "${staged_install}" "${UV_INSTALL_DIR}"
fi

printf '%s  %s\n' "${UV_BINARY_SHA256}" "${UV_BIN}" | sha256sum --check --status

actual_version="$(${UV_BIN} --version)"
if [[ "${actual_version}" != "uv ${UV_VERSION} (x86_64-unknown-linux-gnu)" ]]; then
  printf 'Unexpected uv binary: %s\n' "${actual_version}" >&2
  exit 1
fi

exec "${PROJECT_DIR}/scripts/uv.sh" sync --locked --python /usr/bin/python3.10
