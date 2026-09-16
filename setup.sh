#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./setup.sh check       Check external dependencies and configured paths.
  ./setup.sh help        Show this message.

Setup is intentionally non-destructive. ROS2, vendor SDKs, hardware drivers,
and policy weights must be installed or obtained separately.
EOF
}

case "${1:-check}" in
    check) exec "${SCRIPT_DIR}/check_environment.sh" ;;
    help|-h|--help) usage ;;
    *) echo "Unknown setup command: $1" >&2; usage >&2; exit 2 ;;
esac
