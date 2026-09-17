#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

status=0
ok() { printf '[OK]   %s\n' "$1"; }
warn() { printf '[WARN] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1" >&2; status=1; }
check_path() {
    local label="$1" path="$2"
    if [[ -e "${path}" ]]; then ok "${label}: ${path}"; else fail "${label} not found: ${path}"; fi
}

echo "TRON1 WBC deployment environment"
echo "launcher=${SCRIPT_DIR}"

for command_name in bash git ip ping sudo; do
    command -v "${command_name}" >/dev/null 2>&1 \
        && ok "command ${command_name}" \
        || fail "command ${command_name} is missing"
done

check_path "ROS2 setup" "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
check_path "runtime source" "${TRON_DEPLOY_DIR:-${SCRIPT_DIR}}"
check_path "FAST-LIO root" "${FASTLIO_ROOT:-}"
check_path "Livox workspace" "${LIVOX_WS:-${FASTLIO_ROOT:-}/ros2_ws}/install/setup.bash"
check_path "FAST-LIO workspace" "${FASTLIO_WS:-${FASTLIO_ROOT:-}/fastlio_ws}/install/setup.bash"
check_path "ARX SDK" "${ARX_SDK:-}"
check_path "ARX CAN setup" "${ARX_CAN_SETUP:-}"
check_path "ARX Python" "${ARX_PY:-}"
check_path "Deploy Python" "${DEPLOY_PY:-}"

RUNTIME_DIR="${TRON_DEPLOY_DIR:-${SCRIPT_DIR}}"
check_path "WBC entrypoint" "${RUNTIME_DIR}/deploy_sf_tron1_arm_mujoco.py"
check_path "ground estimator" "${RUNTIME_DIR}/read_ground_height.py"

if [[ -n "${DEPLOY_PY:-}" && -x "${DEPLOY_PY}" ]]; then
    if "${DEPLOY_PY}" -c 'import numpy, scipy, onnxruntime, yaml, rclpy, limxsdk' >/dev/null 2>&1; then
        ok "Deploy Python imports"
    else
        fail "Deploy Python imports (numpy scipy onnxruntime yaml rclpy limxsdk)"
    fi
fi

if [[ -n "${ARX_PY:-}" && -x "${ARX_PY}" ]]; then
    if "${ARX_PY}" -c 'import zmq, click' >/dev/null 2>&1; then
        ok "ARX Python imports"
    else
        fail "ARX Python imports (zmq click)"
    fi
fi

if [[ -n "${POLICY_MODEL_DIR:-}" ]]; then
    for model in actor.onnx contactNet.onnx gru.onnx; do
        check_path "policy ${model}" "${POLICY_MODEL_DIR}/${model}"
    done
else
    warn "POLICY_MODEL_DIR is not set; pass --model-dir when launching."
fi

if ((status == 0)); then
    echo "Environment check passed."
else
    echo "Environment check failed. Fix the items marked [FAIL]." >&2
fi
exit "${status}"
