#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/phi/python__runner"
WBC_ROOT="${ROOT}/tron1-rl-deploy-python"
COMMAND_FILE="${DIFFUSION_WBC_COMMAND_FILE:-/tmp/diffusion_wbc_command.json}"

echo "[bridge] WBC command file: ${COMMAND_FILE}"
echo "[bridge] WBC starts by holding the current EE target."
echo "[bridge] Real outputs remain disabled unless --enable-output is supplied."

exec "${WBC_ROOT}/start_full_wbc_stack.sh" \
    --ee-command-file "${COMMAND_FILE}" \
    --require-diffusion-command \
    --command-ee-frame j6 \
    --hold-current-ee \
    "$@"
