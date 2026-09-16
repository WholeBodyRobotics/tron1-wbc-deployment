# TRON1 WBC Deployment

Clean launchers for the TRON1 + ARX5 whole-body controller stack.

This repository intentionally contains deployment entrypoints only. The WBC
runtime, policy models, FAST-LIO workspaces, LimX SDK, ARX SDK, and ROS2 are
external dependencies configured with environment variables.

## Requirements

- Ubuntu with Bash and ROS2 Humble
- Built Livox ROS2 and FAST-LIO workspaces
- LimX SDK and a working TRON network connection
- ARX SDK, `setup_arx_can.sh`, and the ARX Python environment
- The full runtime source tree containing `deploy_sf_tron1_arm_mujoco.py`
- Policy files: `actor.onnx`, `contactNet.onnx`, and `gru.onnx`

The launcher cannot legally redistribute vendor SDKs or policy weights. Those
components must be obtained from their respective vendors or project owners.

## Configure

Copy the template and edit all absolute paths:

```bash
cp deploy.env.example .env
source .env
export TRON_DEPLOY_DIR TRON_DEPLOY_ROOT FASTLIO_ROOT
export ARX_SDK ARX_CAN_SETUP ARX_PY ARX_ENV DEPLOY_PY
export POLICY_MODEL_DIR
```

`TRON_DEPLOY_DIR` must point to the complete runtime source tree. This keeps
the public launcher repository small and avoids copying private SDKs or policy
weights into it.

Run the non-destructive preflight check:

```bash
./setup.sh check
```

The checker reports missing system tools, ROS2 workspaces, vendor SDKs,
Python imports, runtime files, and policy files. It does not install packages,
change network settings, or move the robot.

## Run

Print available options:

```bash
./deploy.sh --help
```

Run a safe dry-run first. Hardware output stays disabled unless explicitly
enabled:

```bash
./deploy.sh --duration 20
```

Enable real hardware output only after checking the dry-run logs:

```bash
./deploy.sh --enable-output --duration 0
```

Useful modes:

```bash
./deploy.sh --enable-output --gui --duration 0
./start_wbc_diffusion_bridge.sh --duration 0
./start_quest_wbc.sh
```

Logs are written to `runtime_logs/<timestamp>/` in the runtime source tree.
Press `Ctrl+C` to stop the stack and clean up child processes.

## Safety

`--enable-output` can move the real robot. Keep the emergency stop ready,
support the robot during tests, and verify network, joint limits, policy model,
and coordinate frames before enabling output.

## License

MIT. The external runtime, SDKs, ROS2 packages, and policy models retain their
own licenses.
