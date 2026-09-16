#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/phi/python__runner"
DEPLOY_DIR="${ROOT}/tron1-rl-deploy-python"
DEPLOY_PY="/home/phi/miniconda3/envs/deploy/bin/python"
ARX_SDK="${ROOT}/umi-deploy/arx5-sdk"
ARX_CAN_SETUP="${ROOT}/umi-deploy/setup_arx_can.sh"
ARX_PY="/home/phi/miniconda3/envs/arx-py310/bin/python"
ARX_ENV="/home/phi/miniconda3/envs/arx-py310"

ROBOT_IP="${TRON1_IP:-10.192.1.2}"
TRON_IF="${TRON_IF:-enx6c1ff71d2287}"
TRON_SUBNET="${TRON_SUBNET:-10.192.1.0/24}"
META_POLICY_TABLE="${META_POLICY_TABLE:-2022}"
ARM_DURATION=2.0
LEG_DURATION=8.0
HOLD_DURATION=3.0
KP_SCALE=1.0
ENABLE_OUTPUT=false
LOG_DIR="${DEPLOY_DIR}/runtime_logs/reset_$(date +%Y%m%d_%H%M%S)"
ZMQ_PID=""

usage() {
    echo "Usage: $0 --enable-output [options]"
    echo
    echo "Reset the ARX arm and all eight TRON leg joints to the policy initial pose."
    echo "The arm is reset first, followed by the legs; both are then held and checked."
    echo
    echo "Options:"
    echo "  --arm-duration SEC   Arm interpolation duration (default: 2.0)"
    echo "  --leg-duration SEC   Leg interpolation duration (default: 8.0)"
    echo "  --hold SEC           Joint hold/check duration (default: 3.0)"
    echo "  --kp-scale VALUE     Leg Kp scale (default: 1.0)"
    echo "  --robot-ip IP        TRON address (default: 10.192.1.2)"
    echo "  --tron-if IFACE      TRON USB Ethernet interface"
}

while (($#)); do
    case "$1" in
        --enable-output)
            ENABLE_OUTPUT=true
            shift
            ;;
        --arm-duration)
            ARM_DURATION="${2:?--arm-duration requires seconds}"
            shift 2
            ;;
        --leg-duration)
            LEG_DURATION="${2:?--leg-duration requires seconds}"
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
        --tron-if)
            TRON_IF="${2:?--tron-if requires an interface}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! ${ENABLE_OUTPUT}; then
    echo "ERROR: This command physically moves the arm and legs; add --enable-output." >&2
    exit 2
fi

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${ZMQ_PID}" ]] && kill -0 "${ZMQ_PID}" 2>/dev/null; then
        kill -TERM "${ZMQ_PID}" 2>/dev/null || true
        sleep 1
        kill -KILL "${ZMQ_PID}" 2>/dev/null || true
        wait "${ZMQ_PID}" 2>/dev/null || true
    fi
    echo "[reset] Logs: ${LOG_DIR}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

mkdir -p "${LOG_DIR}"

for required in \
    "${DEPLOY_PY}" \
    "${ARX_PY}" \
    "${ARX_CAN_SETUP}" \
    "${ARX_SDK}/python/communication/zmq_server.py" \
    "${DEPLOY_DIR}/reset_robot_initial.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: Required path not found: ${required}" >&2
        exit 1
    fi
done

echo "[1/4] Configuring TRON network on ${TRON_IF}..."
if ! ip link show "${TRON_IF}" >/dev/null 2>&1; then
    echo "ERROR: TRON USB Ethernet interface not found: ${TRON_IF}" >&2
    exit 1
fi
sudo ip link set "${TRON_IF}" up
TRON_HOST_IP="$(
    ip -4 -o addr show dev "${TRON_IF}" scope global |
        awk '$4 ~ /^10[.]192[.]1[.]/ {split($4, a, "/"); print a[1]; exit}'
)"
if [[ -z "${TRON_HOST_IP}" ]]; then
    TRON_HOST_IP="10.192.1.10"
    sudo ip addr replace "${TRON_HOST_IP}/24" dev "${TRON_IF}"
fi
sudo ip route replace "${TRON_SUBNET}" dev "${TRON_IF}" src "${TRON_HOST_IP}"
if ip rule show | grep -q "lookup ${META_POLICY_TABLE}"; then
    sudo ip route replace table "${META_POLICY_TABLE}" \
        "${TRON_SUBNET}" dev "${TRON_IF}" src "${TRON_HOST_IP}"
fi
echo "[network] $(ip route get "${ROBOT_IP}" | head -n 1)"
ping -c 2 -W 1 "${ROBOT_IP}"

echo "[2/4] Configuring ARX CAN..."
pkill -TERM -f "python/communication/zmq_server.py.*L5_umi.*can1" 2>/dev/null || true
sleep 1
"${ARX_CAN_SETUP}" can1

echo "[3/4] Starting ARX ZMQ server..."
(
    cd "${ARX_SDK}"
    exec env \
        PYTHONPATH= \
        LD_LIBRARY_PATH="${ARX_ENV}/lib:/usr/lib/x86_64-linux-gnu" \
        AMENT_PREFIX_PATH="${ARX_ENV}" \
        CMAKE_PREFIX_PATH="${ARX_ENV}" \
        COLCON_PREFIX_PATH= \
        ROS_DISTRO=humble \
        ROS_VERSION=2 \
        ROS_PYTHON_VERSION=3 \
        "${ARX_PY}" python/communication/zmq_server.py \
        --no-gravity-compensation L5_umi can1
) >"${LOG_DIR}/arx_zmq.log" 2>&1 &
ZMQ_PID=$!

ZMQ_READY=false
for _ in $(seq 1 50); do
    if ! kill -0 "${ZMQ_PID}" 2>/dev/null; then
        echo "ERROR: ARX ZMQ server exited during startup." >&2
        tail -n 80 "${LOG_DIR}/arx_zmq.log" >&2 || true
        exit 1
    fi
    if nc -z -w 1 127.0.0.1 8765 >/dev/null 2>&1; then
        ZMQ_READY=true
        break
    fi
    sleep 0.1
done
if ! ${ZMQ_READY}; then
    echo "ERROR: ARX ZMQ port 8765 did not become ready." >&2
    tail -n 80 "${LOG_DIR}/arx_zmq.log" >&2 || true
    exit 1
fi

echo "[4/4] Resetting arm and legs to the policy initial pose..."
echo "[reset] arm_duration=${ARM_DURATION}s leg_duration=${LEG_DURATION}s hold=${HOLD_DURATION}s"
cd "${DEPLOY_DIR}"
"${DEPLOY_PY}" -u reset_robot_initial.py \
    --robot-ip "${ROBOT_IP}" \
    --arx-ip 127.0.0.1 \
    --arx-port 8765 \
    --arm-duration "${ARM_DURATION}" \
    --leg-duration "${LEG_DURATION}" \
    --hold-duration "${HOLD_DURATION}" \
    --leg-kp-scale "${KP_SCALE}" \
    2>&1 | tee "${LOG_DIR}/reset.log"
