#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="/home/phi/python__runner/tron1-rl-deploy-python"
STEAMVR_ROOT="/home/phi/steamvr"
DEPLOY_PY="/home/phi/miniconda3/envs/deploy/bin/python"
QUEST_COMMAND_FILE="/tmp/quest_wbc_command.json"
RUNTIME_LOG_ROOT="${DEPLOY_ROOT}/runtime_logs"
READY_TIMEOUT=180

WBC_PID=""
QUEST_PID=""

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "${QUEST_PID}" ]] && kill -0 "${QUEST_PID}" 2>/dev/null; then
        kill -TERM "${QUEST_PID}" 2>/dev/null || true
        wait "${QUEST_PID}" 2>/dev/null || true
    fi
    if [[ -n "${WBC_PID}" ]] && kill -0 "${WBC_PID}" 2>/dev/null; then
        kill -TERM "${WBC_PID}" 2>/dev/null || true
        wait "${WBC_PID}" 2>/dev/null || true
    fi
    echo "[quest-stack] Stopped Quest and WBC processes started by this run."
}
trap cleanup EXIT INT TERM

if pgrep -f '[q]uest_to_wbc_command.py' >/dev/null; then
    echo "ERROR: An existing quest_to_wbc_command.py process is still running." >&2
    echo "Stop the old Quest terminal before starting this unified stack." >&2
    exit 1
fi
if pgrep -f '[d]eploy_sf_tron1_arm_mujoco.py' >/dev/null; then
    echo "ERROR: An existing real WBC deployment process is still running." >&2
    echo "Stop the old WBC terminal before starting this unified stack." >&2
    exit 1
fi

# A normal Quest bridge exit deliberately latches estop=true. Preserve the old
# command for diagnosis, but never feed it into a new WBC run.
if [[ -e "${QUEST_COMMAND_FILE}" ]]; then
    QUEST_COMMAND_ARCHIVE="/tmp/quest_wbc_command.previous.$(date +%Y%m%d_%H%M%S).$$.json"
    mv -- "${QUEST_COMMAND_FILE}" "${QUEST_COMMAND_ARCHIVE}"
    echo "[quest-stack] Archived previous command: ${QUEST_COMMAND_ARCHIVE}"
fi

mkdir -p "${RUNTIME_LOG_ROOT}"
READY_MARKER="$(mktemp /tmp/start_quest_wbc.ready.XXXXXX)"

echo "[quest-stack] Starting real WBC: deploy6, arm_max_step=0.1, max_leg_step=999"
(
    cd "${DEPLOY_ROOT}"
    DIFFUSION_WBC_COMMAND_FILE="${QUEST_COMMAND_FILE}" \
        exec ./start_wbc_diffusion_bridge.sh \
            --duration 0 \
            --model-dir "${DEPLOY_ROOT}/policy/deploy6/exported" \
            --arm-max-step 0.1 \
            --max-leg-step 999 \
            --enable-output \
            --pre-diffusion-hold-pose 0.2 0 1 0 0 0 \
            --skip-leg-reset
) &
WBC_PID=$!

echo "[quest-stack] Waiting for the WBC diagnostic stream before starting Quest..."
READY_LOG=""
for ((tick = 0; tick < READY_TIMEOUT * 10; tick++)); do
    if ! kill -0 "${WBC_PID}" 2>/dev/null; then
        wait "${WBC_PID}" || true
        echo "ERROR: WBC exited before it became ready." >&2
        exit 1
    fi
    READY_LOG="$({
        find "${RUNTIME_LOG_ROOT}" -type f \
            -name wbc_diagnostics.jsonl -newer "${READY_MARKER}" \
            -size +0c -print 2>/dev/null
    } | sort | tail -n 1)"
    if [[ -n "${READY_LOG}" ]]; then
        break
    fi
    sleep 0.1
done
rm -f -- "${READY_MARKER}"
READY_MARKER=""

if [[ -z "${READY_LOG}" ]]; then
    echo "ERROR: WBC did not create a live diagnostic stream within ${READY_TIMEOUT}s." >&2
    exit 1
fi

echo "[quest-stack] WBC ready: ${READY_LOG}"
echo "[quest-stack] Starting Quest XYZ/RPY control and physical gripper..."
"${DEPLOY_PY}" "${STEAMVR_ROOT}/integration/quest_to_wbc_command.py" \
    --command-file "${QUEST_COMMAND_FILE}" \
    --state-log "${READY_LOG}" \
    --scale 0.3 \
    --max-delta 0.15 \
    --max-angle 0.4 \
    --enable-gripper-output \
    --gripper-port /dev/ttyUSB0 &
QUEST_PID=$!

set +e
wait -n "${WBC_PID}" "${QUEST_PID}"
STACK_STATUS=$?
set -e

if ! kill -0 "${WBC_PID}" 2>/dev/null; then
    echo "[quest-stack] WBC exited; stopping Quest."
else
    echo "[quest-stack] Quest exited; stopping WBC."
fi
exit "${STACK_STATUS}"
