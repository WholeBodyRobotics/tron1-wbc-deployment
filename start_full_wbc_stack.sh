#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TRON_DEPLOY_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
DEPLOY_DIR="${TRON_DEPLOY_DIR:-${SCRIPT_DIR}}"
FASTLIO_ROOT="${FASTLIO_ROOT:-${HOME}/FAST-LIO}"
LIVOX_WS="${LIVOX_WS:-${FASTLIO_ROOT}/ros2_ws}"
FASTLIO_WS="${FASTLIO_WS:-${FASTLIO_ROOT}/fastlio_ws}"
ARX_SDK="${ARX_SDK:-${ROOT}/umi-deploy/arx5-sdk}"
ARX_CAN_SETUP="${ARX_CAN_SETUP:-${ROOT}/umi-deploy/setup_arx_can.sh}"
ARX_PY="${ARX_PY:-${HOME}/miniconda3/envs/arx-py310/bin/python}"
ARX_ENV="${ARX_ENV:-${HOME}/miniconda3/envs/arx-py310}"
DEPLOY_PY="${DEPLOY_PY:-${HOME}/miniconda3/envs/deploy/bin/python}"
LOG_DIR="${DEPLOY_DIR}/runtime_logs/$(date +%Y%m%d_%H%M%S)"

ENABLE_OUTPUT=false
START_GUI=false
SETUP_NETWORK=true
SETUP_CAN=true
ARM_ONLY=false
ARM_HOME_ONLY=false
ARM_JOINT_TEST=""
ARM_TEST_DELTA=0.02
HOLD_CURRENT_EE=false
PRE_DIFFUSION_HOLD_POSITION=()
PRE_DIFFUSION_HOLD_POSE=()
SKIP_LEG_RESET=false
ARM_MAX_STEP=0
MAX_LEG_STEP=0
SE3_DECAY_RATE=0.5
DURATION=20
COMMAND=(0 0 0.8 0 0 0)
COMMAND_FRAME="world"
COMMAND_EE_FRAME="j6"
POLICY_EE_FRAME="j6"
MODEL_DIR=""
EE_COMMAND_FILE=""
REQUIRE_DIFFUSION_COMMAND=false
FREEZE_WORLD_BASE=false
KEYBOARD_COMMAND=false
KEYBOARD_STEP=0.1
GEN_GRIPPER=false
GRIPPER_PORT="/dev/ttyUSB0"
GRIPPER_SDK_ROOT="/home/phi/python__runner/umi-deploy/gen_con_sdk_python_release"
GRIPPER_ENCODER_FREQUENCY=30
GRIPPER_FEEDBACK_TIMEOUT=3
GRIPPER_INITIAL_WIDTH=0.08
GRIPPER_STEP=0.01
BRIDGE_COMMAND_TIMEOUT=2.0
FIXED_GROUND_HEIGHT=""
TRON_ROBOT_IP="${TRON1_IP:-10.192.1.2}"
TRON_IF="${TRON_IF:-enx6c1ff71d2287}"
TRON_FALLBACK_CIDR="${TRON_HOST_CIDR:-10.192.1.10/24}"
TRON_SUBNET="${TRON_SUBNET:-10.192.1.0/24}"
META_POLICY_TABLE="${META_POLICY_TABLE:-2022}"

usage() {
    echo "Usage: $0 [--enable-output] [--gui] [--keyboard-command] [--keyboard-step M] [--gen-gripper] [--gripper-port DEV] [--ee-command-file FILE] [--require-diffusion-command] [--pre-diffusion-hold-position X Y Z | --pre-diffusion-hold-pose X Y Z R P Y] [--freeze-world-base] [--fixed-ground-height M] [--bridge-command-timeout SEC] [--duration SEC] [--command X Y Z R P Y] [--command-frame world|base] [--command-ee-frame eef_link|j6] [--policy-ee-frame eef_link|j6] [--model-dir DIR] [--se3-decay-rate RATE] [--arm-max-step RAD] [--max-leg-step RAD] [--arm-only] [--arm-home-only] [--arm-joint-test J1..J6] [--arm-test-delta RAD] [--hold-current-ee] [--skip-leg-reset] [--skip-network] [--skip-can]"
    echo
    echo "Without --enable-output, the policy runs without sending real policy commands."
    echo "--arm-only disables every leg command, including leg reset."
    echo "--arm-home-only moves only the arm to its configured home and exits."
    echo "--skip-leg-reset skips leg homing and its 0.1 rad check; use only after running reset_tron_legs.sh."
    echo "--arm-max-step limits each arm target change per 50 Hz policy step (default: 0, disabled)."
    echo "--max-leg-step limits each leg target change per 50 Hz policy step (default: 0, disabled)."
    echo "--se3-decay-rate sets SE(3) reference decay per second (default: 0.5)."
    echo "--model-dir selects a directory containing actor.onnx, contactNet.onnx, and gru.onnx."
    echo "--ee-command-file reads an external teleop/Diffusion command JSON."
    echo "--gui opens a live EE/base pose monitor and XYZ/RPY target panel."
    echo "--keyboard-command enables W/S=+/-X, A/D=+/-Y, R/F=+/-Z without Enter."
    echo "--keyboard-step sets the XYZ increment per key press in metres (default: 0.1)."
    echo "--gen-gripper enables GenRobot keyboard control: T=open increment, G=close increment."
    echo "--gripper-step sets the T/G increment in metres (default: 0.01)."
    echo "--gripper-port selects the GenRobot serial device (default: /dev/ttyUSB0)."
    echo "--command-frame selects whether XYZ/RPY is expressed in world or robot base."
    echo "--command-ee-frame selects the command EE frame (default: j6)."
    echo "--policy-ee-frame selects the policy training EE frame (default: j6)."
    echo "--require-diffusion-command blocks output until fresh chunks arrive and adds a watchdog."
    echo "--freeze-world-base freezes the startup base pose for a physically stationary robot."
    echo "--fixed-ground-height bypasses floor-plane fitting and publishes a fixed height."
    echo "Ground fitting waits for a 5 mm stable window, stays live through startup, then freezes at output start."
}

while (($#)); do
    case "$1" in
        --enable-output)
            ENABLE_OUTPUT=true
            shift
            ;;
        --gui)
            START_GUI=true
            shift
            ;;
        --keyboard-command)
            KEYBOARD_COMMAND=true
            shift
            ;;
        --keyboard-step)
            KEYBOARD_STEP="${2:?--keyboard-step requires metres per key press}"
            shift 2
            ;;
        --gen-gripper)
            GEN_GRIPPER=true
            shift
            ;;
        --gripper-port)
            GRIPPER_PORT="${2:?--gripper-port requires a serial device}"
            shift 2
            ;;
        --gripper-sdk-root)
            GRIPPER_SDK_ROOT="${2:?--gripper-sdk-root requires a directory}"
            shift 2
            ;;
        --gripper-encoder-frequency)
            GRIPPER_ENCODER_FREQUENCY="${2:?--gripper-encoder-frequency requires Hz}"
            shift 2
            ;;
        --gripper-feedback-timeout)
            GRIPPER_FEEDBACK_TIMEOUT="${2:?--gripper-feedback-timeout requires seconds}"
            shift 2
            ;;
        --gripper-initial-width)
            GRIPPER_INITIAL_WIDTH="${2:?--gripper-initial-width requires metres}"
            shift 2
            ;;
        --gripper-step)
            GRIPPER_STEP="${2:?--gripper-step requires metres per key press}"
            shift 2
            ;;
        --arm-only)
            ARM_ONLY=true
            shift
            ;;
        --arm-home-only)
            ARM_HOME_ONLY=true
            ARM_ONLY=true
            shift
            ;;
        --arm-joint-test)
            ARM_JOINT_TEST="${2:?--arm-joint-test requires J1, J2, J3, J4, J5, or J6}"
            ARM_ONLY=true
            shift 2
            ;;
        --arm-test-delta)
            ARM_TEST_DELTA="${2:?--arm-test-delta requires radians}"
            shift 2
            ;;
        --hold-current-ee)
            HOLD_CURRENT_EE=true
            shift
            ;;
        --pre-diffusion-hold-position)
            PRE_DIFFUSION_HOLD_POSITION=(
                "${2:?--pre-diffusion-hold-position requires X Y Z}"
                "${3:?--pre-diffusion-hold-position requires X Y Z}"
                "${4:?--pre-diffusion-hold-position requires X Y Z}"
            )
            shift 4
            ;;
        --pre-diffusion-hold-pose)
            PRE_DIFFUSION_HOLD_POSE=(
                "${2:?--pre-diffusion-hold-pose requires X Y Z R P Y}"
                "${3:?--pre-diffusion-hold-pose requires X Y Z R P Y}"
                "${4:?--pre-diffusion-hold-pose requires X Y Z R P Y}"
                "${5:?--pre-diffusion-hold-pose requires X Y Z R P Y}"
                "${6:?--pre-diffusion-hold-pose requires X Y Z R P Y}"
                "${7:?--pre-diffusion-hold-pose requires X Y Z R P Y}"
            )
            shift 7
            ;;
        --skip-leg-reset)
            SKIP_LEG_RESET=true
            shift
            ;;
        --max-leg-step)
            MAX_LEG_STEP="${2:?--max-leg-step requires radians per policy step}"
            shift 2
            ;;
        --arm-max-step|--arm_max_step)
            ARM_MAX_STEP="${2:?--arm-max-step requires radians per policy step}"
            shift 2
            ;;
        --se3-decay-rate)
            SE3_DECAY_RATE="${2:?--se3-decay-rate requires a value per second}"
            shift 2
            ;;
        --duration)
            DURATION="${2:?--duration requires seconds}"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="${2:?--model-dir requires a policy directory}"
            shift 2
            ;;
        --ee-command-file)
            EE_COMMAND_FILE="${2:?--ee-command-file requires a path}"
            shift 2
            ;;
        --require-diffusion-command)
            REQUIRE_DIFFUSION_COMMAND=true
            shift
            ;;
        --freeze-world-base)
            FREEZE_WORLD_BASE=true
            shift
            ;;
        --bridge-command-timeout)
            BRIDGE_COMMAND_TIMEOUT="${2:?--bridge-command-timeout requires seconds}"
            shift 2
            ;;
        --fixed-ground-height)
            FIXED_GROUND_HEIGHT="${2:?--fixed-ground-height requires meters}"
            shift 2
            ;;
        --command)
            if (($# < 7)); then
                echo "ERROR: --command requires 6 values: X Y Z roll pitch yaw" >&2
                exit 2
            fi
            COMMAND=("$2" "$3" "$4" "$5" "$6" "$7")
            shift 7
            ;;
        --command-frame)
            COMMAND_FRAME="${2:?--command-frame requires world or base}"
            if [[ "${COMMAND_FRAME}" != "world" && "${COMMAND_FRAME}" != "base" ]]; then
                echo "ERROR: --command-frame must be world or base" >&2
                exit 2
            fi
            shift 2
            ;;
        --command-ee-frame)
            COMMAND_EE_FRAME="${2:?--command-ee-frame requires eef_link or j6}"
            if [[ "${COMMAND_EE_FRAME}" != "eef_link" && "${COMMAND_EE_FRAME}" != "j6" ]]; then
                echo "ERROR: --command-ee-frame must be eef_link or j6" >&2
                exit 2
            fi
            shift 2
            ;;
        --policy-ee-frame)
            POLICY_EE_FRAME="${2:?--policy-ee-frame requires eef_link or j6}"
            if [[ "${POLICY_EE_FRAME}" != "eef_link" && "${POLICY_EE_FRAME}" != "j6" ]]; then
                echo "ERROR: --policy-ee-frame must be eef_link or j6" >&2
                exit 2
            fi
            shift 2
            ;;
        --skip-network)
            SETUP_NETWORK=false
            shift
            ;;
        --skip-can)
            SETUP_CAN=false
            shift
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

if ${ARM_HOME_ONLY} && ! ${ENABLE_OUTPUT}; then
    echo "ERROR: --arm-home-only physically moves the arm; add --enable-output to confirm." >&2
    exit 2
fi
if [[ -n "${ARM_JOINT_TEST}" ]] && ! ${ENABLE_OUTPUT}; then
    echo "ERROR: --arm-joint-test physically moves the arm; add --enable-output to confirm." >&2
    exit 2
fi
if ${START_GUI} && ${HOLD_CURRENT_EE}; then
    echo "ERROR: Do not combine --gui with --hold-current-ee; the GUI needs an explicit --command initial pose." >&2
    exit 2
fi
if ${KEYBOARD_COMMAND} && ${REQUIRE_DIFFUSION_COMMAND}; then
    echo "ERROR: --keyboard-command cannot be combined with --require-diffusion-command." >&2
    exit 2
fi
if ((${#PRE_DIFFUSION_HOLD_POSITION[@]})) && ((${#PRE_DIFFUSION_HOLD_POSE[@]})); then
    echo "ERROR: Choose only one pre-Diffusion hold target." >&2
    exit 2
fi
if ${GEN_GRIPPER} && ! ${KEYBOARD_COMMAND}; then
    echo "ERROR: --gen-gripper requires --keyboard-command." >&2
    exit 2
fi
if ${REQUIRE_DIFFUSION_COMMAND} && [[ "${COMMAND_EE_FRAME}" != "j6" ]]; then
    echo "ERROR: Diffusion bridge commands are already J6/link6; use --command-ee-frame j6." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

LIVOX_PID=""
FASTLIO_PID=""
GROUND_HEIGHT_PID=""
ZMQ_PID=""
GUI_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    echo
    echo "[cleanup] Stopping processes started by this run..."
    for pid in "${GUI_PID}" "${ZMQ_PID}" "${GROUND_HEIGHT_PID}" "${FASTLIO_PID}" "${LIVOX_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    # Livox SDK can hang while shutting down. Do not leave its child process
    # bound to 56000/56101/56201/56301/56401 for the next launch.
    sleep 2
    pkill -KILL -x livox_ros_drive 2>/dev/null || true
    pkill -KILL -x fastlio_mapping 2>/dev/null || true
    for pid in "${GUI_PID}" "${ZMQ_PID}" "${GROUND_HEIGHT_PID}" "${FASTLIO_PID}" "${LIVOX_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    wait "${GUI_PID}" "${ZMQ_PID}" "${GROUND_HEIGHT_PID}" "${FASTLIO_PID}" "${LIVOX_PID}" 2>/dev/null || true
    echo "[cleanup] Logs: ${LOG_DIR}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: Required file not found: $1" >&2
        exit 1
    fi
}

wait_for_message() {
    local topic="$1"
    local timeout_s="$2"
    local label="$3"
    local start_s=${SECONDS}
    local elapsed_s
    local remaining_s

    echo "[wait] ${label}: ${topic}"

    # `ros2 topic echo` exits immediately when the topic is not registered yet,
    # so first wait for ROS discovery instead of wrapping that first call in
    # timeout and accidentally treating it as a real wait.
    while ! ros2 topic list 2>/dev/null | grep -Fxq "${topic}"; do
        elapsed_s=$((SECONDS - start_s))
        if ((elapsed_s >= timeout_s)); then
            echo "ERROR: Topic ${topic} did not appear within ${timeout_s}s" >&2
            return 1
        fi
        sleep 1
    done

    elapsed_s=$((SECONDS - start_s))
    remaining_s=$((timeout_s - elapsed_s))
    if ((remaining_s < 1)); then
        remaining_s=1
    fi
    if ! timeout "${remaining_s}" ros2 topic echo "${topic}" --once >/dev/null 2>&1; then
        echo "ERROR: No message received on ${topic} within ${timeout_s}s" >&2
        return 1
    fi
    echo "[ok] ${label}"
}

require_file "/opt/ros/humble/setup.bash"
require_file "${LIVOX_WS}/install/setup.bash"
require_file "${FASTLIO_WS}/install/setup.bash"
require_file "${ARX_CAN_SETUP}"
require_file "${ARX_SDK}/python/communication/zmq_server.py"
require_file "${DEPLOY_DIR}/deploy_sf_tron1_arm_mujoco.py"
require_file "${DEPLOY_DIR}/read_ground_height.py"
require_file "${DEPLOY_DIR}/livox_mid360_stable.launch.py"
require_file "${DEPLOY_DIR}/wait_for_stable_odom.py"

# ROS 2 generated setup scripts may probe environment variables that are not
# defined in a fresh shell.  Temporarily disable nounset only while sourcing
# them, then restore this script's strict mode.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${LIVOX_WS}/install/setup.bash"
# shellcheck disable=SC1091
source "${FASTLIO_WS}/install/setup.bash"
set -u

echo "[preflight] Checking Python environments..."
"${ARX_PY}" -c "import zmq, click" >/dev/null
"${DEPLOY_PY}" -c "import scipy, onnxruntime, yaml, rclpy, limxsdk.datatypes" >/dev/null

if ${SETUP_NETWORK}; then
    echo "[network] Configuring enp4s0 for MID360..."
    sudo ip link set enp4s0 up
    sudo ip addr replace 192.168.1.50/24 dev enp4s0
    # Remove the temporary TRON address installed by an older version of this
    # script.  TRON is physically attached through its USB Ethernet adapter.
    if ip -4 -o addr show dev enp4s0 | grep -Fq "10.192.1.10/24"; then
        sudo ip addr del 10.192.1.10/24 dev enp4s0
    fi
    ip addr show dev enp4s0
    if ! ping -c 2 -W 1 192.168.1.161; then
        echo "ERROR: MID360 at 192.168.1.161 is unreachable" >&2
        exit 1
    fi

    echo "[network] Configuring ${TRON_IF} for TRON..."
    if ! ip link show "${TRON_IF}" >/dev/null 2>&1; then
        echo "ERROR: TRON USB Ethernet interface ${TRON_IF} was not found." >&2
        echo "Reconnect the TRON USB network cable or set TRON_IF explicitly." >&2
        exit 1
    fi
    sudo ip link set "${TRON_IF}" up
    TRON_HOST_IP="$(
        ip -4 -o addr show dev "${TRON_IF}" scope global |
            awk '$4 ~ /^10[.]192[.]1[.]/ {split($4, a, "/"); print a[1]; exit}'
    )"
    if [[ -z "${TRON_HOST_IP}" ]]; then
        sudo ip addr replace "${TRON_FALLBACK_CIDR}" dev "${TRON_IF}"
        TRON_HOST_IP="${TRON_FALLBACK_CIDR%/*}"
    fi
    sudo ip route replace "${TRON_SUBNET}" dev "${TRON_IF}" src "${TRON_HOST_IP}"
    # The local Meta/TUN service installs an early policy rule whose default
    # route otherwise captures 10.192.1.2.  Add the physical TRON subnet to
    # that table as a more-specific route instead of disabling the service.
    if ip rule show | grep -q "lookup ${META_POLICY_TABLE}"; then
        sudo ip route replace table "${META_POLICY_TABLE}" \
            "${TRON_SUBNET}" dev "${TRON_IF}" src "${TRON_HOST_IP}"
    fi
    ip addr show dev "${TRON_IF}"
    echo "[network] Route to TRON: $(ip route get "${TRON_ROBOT_IP}" | head -n 1)"
    if ! ping -c 2 -W 1 "${TRON_ROBOT_IP}"; then
        echo "ERROR: TRON at ${TRON_ROBOT_IP} is unreachable on ${TRON_IF}." >&2
        echo "Check the TRON USB Ethernet cable and robot power." >&2
        exit 1
    fi
fi

echo "[restart] Stopping stale radar, FAST-LIO, ground-reference and ZMQ processes..."
pkill -TERM -x livox_ros_drive 2>/dev/null || true
pkill -TERM -x fastlio_mapping 2>/dev/null || true
pkill -TERM -f "read_ground_height.py" 2>/dev/null || true
pkill -TERM -f "python/communication/zmq_server.py.*L5_umi.*can1" 2>/dev/null || true
sleep 2
# The Livox SDK has been observed to ignore SIGTERM when its data channel is
# stuck. Force-remove any survivor before opening the fixed UDP ports again.
pkill -KILL -x livox_ros_drive 2>/dev/null || true
pkill -KILL -x fastlio_mapping 2>/dev/null || true
pkill -KILL -f "read_ground_height.py" 2>/dev/null || true
sleep 1
if pgrep -x livox_ros_drive >/dev/null; then
    echo "ERROR: A stale Livox driver is still running; refusing to start a duplicate." >&2
    exit 1
fi

echo "[1/4] Starting MID360 driver..."
LIVOX_READY=false
for attempt in 1 2; do
    echo "[1/4] MID360 attempt ${attempt}/2"
    (
        cd "${LIVOX_WS}"
        exec ros2 launch "${DEPLOY_DIR}/livox_mid360_stable.launch.py"
    ) >>"${LOG_DIR}/livox.log" 2>&1 &
    LIVOX_PID=$!

    if wait_for_message /livox/lidar 30 "MID360 lidar"; then
        LIVOX_READY=true
        break
    fi

    echo "[retry] MID360 did not publish; restarting its driver..."
    kill -TERM "${LIVOX_PID}" 2>/dev/null || true
    wait "${LIVOX_PID}" 2>/dev/null || true
    LIVOX_PID=""
    sleep 3
done

if ! ${LIVOX_READY}; then
    echo "ERROR: MID360 stayed reachable by ping but published no point cloud." >&2
    echo "----- ${LOG_DIR}/livox.log (last 80 lines) -----" >&2
    tail -n 80 "${LOG_DIR}/livox.log" >&2 || true
    exit 1
fi
wait_for_message /livox/imu 15 "MID360 IMU"

FASTLIO_STABLE=false
for attempt in 1 2; do
    if ((attempt == 2)); then
        echo "[retry] Stale FAST-LIO pipeline detected; restarting MID360 and FAST-LIO..."

        if [[ -n "${FASTLIO_PID}" ]]; then
            kill -TERM "${FASTLIO_PID}" 2>/dev/null || true
        fi
        if [[ -n "${LIVOX_PID}" ]]; then
            kill -TERM "${LIVOX_PID}" 2>/dev/null || true
        fi
        sleep 2
        pkill -KILL -x livox_ros_drive 2>/dev/null || true
        pkill -KILL -x fastlio_mapping 2>/dev/null || true
        kill -KILL "${FASTLIO_PID}" "${LIVOX_PID}" 2>/dev/null || true
        wait "${FASTLIO_PID}" "${LIVOX_PID}" 2>/dev/null || true
        FASTLIO_PID=""
        LIVOX_PID=""
        sleep 1

        echo "[retry] Starting a fresh MID360 stream for FAST-LIO retry..."
        (
            cd "${LIVOX_WS}"
            exec ros2 launch "${DEPLOY_DIR}/livox_mid360_stable.launch.py"
        ) >>"${LOG_DIR}/livox.log" 2>&1 &
        LIVOX_PID=$!
        wait_for_message /livox/lidar 30 "MID360 lidar retry"
        wait_for_message /livox/imu 15 "MID360 IMU retry"
    fi

    echo "[2/4] Starting FAST-LIO attempt ${attempt}/2 with a fresh origin..."
    echo "===== FAST-LIO attempt ${attempt}/2 =====" >>"${LOG_DIR}/fastlio.log"
    (
        cd "${FASTLIO_WS}"
        exec ros2 launch fast_lio mapping.launch.py \
            config_path:="${DEPLOY_DIR}" \
            config_file:=fastlio_mid360_wbc.yaml \
            rviz:=false
    ) >>"${LOG_DIR}/fastlio.log" 2>&1 &
    FASTLIO_PID=$!
    wait_for_message /Odometry 20 "FAST-LIO odometry attempt ${attempt}/2"

    if ((attempt == 1)); then
        ODOM_STABILITY_TIMEOUT=20
    else
        ODOM_STABILITY_TIMEOUT=35
    fi
    echo "[2/4] Checking fresh, stable FAST-LIO odometry (attempt ${attempt}/2)..."
    echo "===== odometry stability attempt ${attempt}/2 =====" \
        >>"${LOG_DIR}/odom_stability.log"
    if /usr/bin/python3 -u "${DEPLOY_DIR}/wait_for_stable_odom.py" \
        --topic /Odometry \
        --samples 100 \
        --timeout "${ODOM_STABILITY_TIMEOUT}" \
        --max-position-span 0.01 \
        --max-angle-span 0.01 \
        --max-source-age 0.25 \
        --max-receive-gap 0.30 \
        --min-window-duration 8.0 \
        >>"${LOG_DIR}/odom_stability.log" 2>&1; then
        FASTLIO_STABLE=true
        break
    fi

    echo "[retry] FAST-LIO attempt ${attempt}/2 did not produce fresh stable odometry." >&2
    tail -n 20 "${LOG_DIR}/odom_stability.log" >&2 || true
done

if ! ${FASTLIO_STABLE}; then
    echo "ERROR: FAST-LIO remained stale or unstable after a full sensor restart." >&2
    tail -n 80 "${LOG_DIR}/odom_stability.log" >&2 || true
    exit 1
fi
tail -n 1 "${LOG_DIR}/odom_stability.log"

if ${SETUP_CAN}; then
    echo "[3/4] Configuring ARX CAN..."
    "${ARX_CAN_SETUP}" can1
else
    echo "[3/4] Skipping CAN setup; expecting can1 to be ready."
fi

echo "[3/4] Starting ARX ZMQ server without gravity compensation..."
(
    cd "${ARX_SDK}"
    # arx-py310 contains its own ROS/URDF libraries. Do not mix them with the
    # system ROS libraries sourced above for Livox and FAST-LIO.
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
sleep 2
if ! kill -0 "${ZMQ_PID}" 2>/dev/null; then
    echo "ERROR: ARX ZMQ server exited; see ${LOG_DIR}/arx_zmq.log" >&2
    exit 1
fi

echo "[3/4] Measuring a settled ground height and pairing it with FAST-LIO..."
GROUND_HEIGHT_ARGS=()
if [[ -n "${FIXED_GROUND_HEIGHT}" ]]; then
    GROUND_HEIGHT_ARGS+=(--fixed-height "${FIXED_GROUND_HEIGHT}")
    echo "[3/4] Using fixed ground height: ${FIXED_GROUND_HEIGHT} m"
fi
(
    cd "${DEPLOY_DIR}"
    exec /usr/bin/python3 -u read_ground_height.py "${GROUND_HEIGHT_ARGS[@]}"
) >"${LOG_DIR}/ground_height.log" 2>&1 &
GROUND_HEIGHT_PID=$!
if ! wait_for_message /ground_height_reference 45 \
    "settled ground height + FAST-LIO reference"; then
    echo "----- ${LOG_DIR}/ground_height.log (last 120 lines) -----" >&2
    tail -n 120 "${LOG_DIR}/ground_height.log" >&2 || true
    exit 1
fi
wait_for_message /ground_height 5 "paired MID360 ground height"

echo "[4/4] Starting WBC deployment (duration=${DURATION}s, output=${ENABLE_OUTPUT})..."
DEPLOY_ARGS=(
    --use-ros2-odom
    --ground-height-topic /ground_height
    --ground-reference-topic /ground_height_reference
    --ground-freeze-topic /ground_height_freeze
    --lidar-to-base-xyz -0.14 0 0.0677
    --fastlio-lidar-to-imu-xyz -0.011 -0.02329 0.04412
    --duration "${DURATION}"
    --command "${COMMAND[@]}"
    --command-frame "${COMMAND_FRAME}"
    --command-ee-frame "${COMMAND_EE_FRAME}"
    --policy-ee-frame "${POLICY_EE_FRAME}"
    --arm-max-step "${ARM_MAX_STEP}"
    --max-leg-step "${MAX_LEG_STEP}"
    --leg-kp-scale 1
    --action-smoothing 0
    --se3-decay-rate "${SE3_DECAY_RATE}"
    --diagnostic-log "${LOG_DIR}/wbc_diagnostics.jsonl"
    --max-ee-position-regression 99
    --max-se3-regression 999
    --max-arm-track-error 999
    --max-leg-track-error 999
)
if ${KEYBOARD_COMMAND}; then
    DEPLOY_ARGS+=(--keyboard-command --keyboard-step "${KEYBOARD_STEP}")
fi
if ${GEN_GRIPPER}; then
    DEPLOY_ARGS+=(
        --gen-gripper
        --gripper-port "${GRIPPER_PORT}"
        --gripper-sdk-root "${GRIPPER_SDK_ROOT}"
        --gripper-encoder-frequency "${GRIPPER_ENCODER_FREQUENCY}"
        --gripper-feedback-timeout "${GRIPPER_FEEDBACK_TIMEOUT}"
        --gripper-initial-width "${GRIPPER_INITIAL_WIDTH}"
        --gripper-step "${GRIPPER_STEP}"
    )
fi
if [[ -n "${MODEL_DIR}" ]]; then
    for model_file in actor.onnx contactNet.onnx gru.onnx; do
        if [[ ! -f "${MODEL_DIR}/${model_file}" ]]; then
            echo "ERROR: Missing policy model: ${MODEL_DIR}/${model_file}" >&2
            exit 2
        fi
    done
    DEPLOY_ARGS+=(--model-dir "${MODEL_DIR}")
    echo "[policy] Using model directory: ${MODEL_DIR}"
fi
if ${REQUIRE_DIFFUSION_COMMAND}; then
    if [[ -z "${EE_COMMAND_FILE}" ]]; then
        echo "ERROR: --require-diffusion-command requires --ee-command-file." >&2
        exit 2
    fi
    DEPLOY_ARGS+=(
        --require-diffusion-command
        --bridge-command-timeout "${BRIDGE_COMMAND_TIMEOUT}"
    )
fi
if ${FREEZE_WORLD_BASE}; then
    DEPLOY_ARGS+=(--freeze-world-base)
fi
if ${ENABLE_OUTPUT}; then
    DEPLOY_ARGS+=(--enable-output)
else
    echo "[safety] --enable-output was not supplied; no real policy commands will be sent."
fi
if ${ARM_ONLY}; then
    DEPLOY_ARGS+=(--no-legs)
    echo "[safety] Arm-only mode: no leg reset or leg policy commands will be sent."
fi
if ${ARM_HOME_ONLY}; then
    DEPLOY_ARGS+=(--arm-home-only)
fi
if [[ -n "${ARM_JOINT_TEST}" ]]; then
    DEPLOY_ARGS+=(--arm-joint-test "${ARM_JOINT_TEST}" --arm-test-delta "${ARM_TEST_DELTA}")
fi
if ${HOLD_CURRENT_EE}; then
    DEPLOY_ARGS+=(--hold-current-ee)
fi
if ((${#PRE_DIFFUSION_HOLD_POSITION[@]})); then
    DEPLOY_ARGS+=(
        --pre-diffusion-hold-position
        "${PRE_DIFFUSION_HOLD_POSITION[@]}"
    )
fi
if ((${#PRE_DIFFUSION_HOLD_POSE[@]})); then
    DEPLOY_ARGS+=(
        --pre-diffusion-hold-pose
        "${PRE_DIFFUSION_HOLD_POSE[@]}"
    )
fi
if ${SKIP_LEG_RESET}; then
    DEPLOY_ARGS+=(--skip-leg-reset)
    echo "[safety] Leg reset/home check skipped by explicit request."
fi

if ${START_GUI}; then
    if [[ -n "${EE_COMMAND_FILE}" ]]; then
        echo "ERROR: --gui and --ee-command-file are mutually exclusive." >&2
        exit 2
    fi
    CONTROL_FILE="${LOG_DIR}/ee_command.json"
    echo "[gui] Starting XYZ/RPY panel; control file=${CONTROL_FILE}"
    /usr/bin/python3 "${DEPLOY_DIR}/ee_command_panel.py" \
        --out "${CONTROL_FILE}" \
        --state-log "${LOG_DIR}/wbc_diagnostics.jsonl" \
        --pose "${COMMAND[*]}" \
        --command-frame "${COMMAND_FRAME}" \
        --ee-frame "${COMMAND_EE_FRAME}" \
        >"${LOG_DIR}/gui.log" 2>&1 &
    GUI_PID=$!
    DEPLOY_ARGS+=(--ee-command-file "${CONTROL_FILE}")
elif [[ -n "${EE_COMMAND_FILE}" ]]; then
    CONTROL_FILE="$(readlink -m "${EE_COMMAND_FILE}")"
    mkdir -p "$(dirname "${CONTROL_FILE}")"
    echo "[command] External command file=${CONTROL_FILE}"
    DEPLOY_ARGS+=(--ee-command-file "${CONTROL_FILE}")
fi

cd "${DEPLOY_DIR}"
"${DEPLOY_PY}" -u deploy_sf_tron1_arm_mujoco.py "${DEPLOY_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/deploy.log"

if ! ${ENABLE_OUTPUT}; then
    echo "[dry-run] Analyzing 50 Hz CPU performance..."
    "${DEPLOY_PY}" "${DEPLOY_DIR}/analyze_wbc_performance.py" \
        "${LOG_DIR}/wbc_diagnostics.jsonl"
fi
