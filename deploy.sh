#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${SCRIPT_DIR}/start_full_wbc_stack.sh"

usage() {
    cat <<'EOF'
Usage:
  ./deploy.sh [options]

This is the public entry point for the real WBC deployment.
It forwards options to start_full_wbc_stack.sh.

Safe examples:
  ./deploy.sh --help
  ./deploy.sh --duration 20
  ./deploy.sh --enable-output --duration 0
  ./deploy.sh --enable-output --gui --duration 0

Environment overrides:
  TRON_DEPLOY_ROOT, FASTLIO_ROOT, LIVOX_WS, FASTLIO_WS, ARX_SDK,
  ARX_CAN_SETUP, ARX_PY, ARX_ENV, DEPLOY_PY
EOF
}

if [[ ! -x "${MAIN}" ]]; then
    echo "ERROR: deployment entry point is not executable: ${MAIN}" >&2
    exit 1
fi

if (($# == 0)); then
    echo "[deploy] Starting safe dry-run; add --enable-output for hardware commands."
fi

exec "${MAIN}" "$@"
