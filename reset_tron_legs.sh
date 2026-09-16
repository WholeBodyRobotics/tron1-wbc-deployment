#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_DIR="/home/phi/python__runner/tron1-rl-deploy-python"
DEPLOY_PY="/home/phi/miniconda3/envs/deploy/bin/python"
ROBOT_IP="${TRON1_IP:-10.192.1.2}"
RESET_DURATION=5.0
HOLD_DURATION=2.0
KP_SCALE=0.5
ENABLE_OUTPUT=false

usage() {
    echo "Usage: $0 --enable-output [--duration SEC] [--hold SEC] [--kp-scale VALUE] [--robot-ip IP]"
    echo "Resets only the eight TRON leg joints; no radar, FAST-LIO, ARX arm, or policy is started."
}

while (($#)); do
    case "$1" in
        --enable-output)
            ENABLE_OUTPUT=true
            shift
            ;;
        --duration)
            RESET_DURATION="${2:?--duration requires seconds}"
            shift 2
            ;;
        --hold)
            HOLD_DURATION="${2:?--hold requires seconds}"
            shift 2
            ;;
        --kp-scale)
            KP_SCALE="${2:?--kp-scale requires a value}"
            shift 2
            ;;
        --robot-ip)
            ROBOT_IP="${2:?--robot-ip requires an address}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! ${ENABLE_OUTPUT}; then
    echo "ERROR: This command physically moves both legs; add --enable-output to confirm." >&2
    exit 2
fi
if [[ ! -x "${DEPLOY_PY}" ]]; then
    echo "ERROR: Deployment Python not found: ${DEPLOY_PY}" >&2
    exit 1
fi

echo "[TRON leg reset] Keep the robot supported and keep the emergency stop ready."
echo "[TRON leg reset] target=training home (all 8 leg joints at 0 rad)"
echo "[TRON leg reset] duration=${RESET_DURATION}s hold=${HOLD_DURATION}s kp_scale=${KP_SCALE}"

cd "${DEPLOY_DIR}"
exec "${DEPLOY_PY}" deploy_sf_tron1_arm_mujoco.py \
    --robot-ip "${ROBOT_IP}" \
    --no-arm \
    --enable-output \
    --leg-home-only \
    --leg-reset-duration "${RESET_DURATION}" \
    --leg-test-hold "${HOLD_DURATION}" \
    --leg-kp-scale "${KP_SCALE}" \
    --leg-publish-rate 500
