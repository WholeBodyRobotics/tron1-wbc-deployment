# TRON1 WBC Deployment

Deployment entrypoints and public runtime code for the TRON1 + ARX5 whole-body controller stack.

The WBC Python runtime, launch files, configuration, and the required robot
URDF are included. Policy models, FAST-LIO workspaces, LimX SDK, ARX SDK, and
ROS2 remain external dependencies configured with environment variables.

## Requirements

- Ubuntu with Bash and ROS2 Humble
- Built Livox ROS2 and FAST-LIO workspaces
- LimX SDK and a working TRON network connection
- ARX SDK, `setup_arx_can.sh`, and the ARX Python environment
- The included runtime source tree containing `deploy_sf_tron1_arm_mujoco.py`
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

By default the launcher uses this repository as `TRON_DEPLOY_DIR`. Set
`TRON_DEPLOY_DIR` only when using a separate runtime checkout.

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

## External components

The repository does not redistribute vendor SDKs, hardware drivers, or policy
weights. Obtain those components from their respective vendors or project
owners and configure their paths in `.env`.

## License

MIT. The external runtime, SDKs, ROS2 packages, and policy models retain their
own licenses.
