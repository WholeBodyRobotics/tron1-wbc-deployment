# Policy archive

All newly exported policies should be stored under this directory in a separate,
uniquely named subdirectory. Keep the export metadata or training checkpoint next
to its `exported/` directory so that the policy remains traceable.

Current policies:

- `deploy1`: model 12600, trained on 2026-07-18 (see `note.txt`).
- `deploy2`: GPU 1 run from 2026-07-19; includes `model_8000.pt` and reward notes.
- `deploy3`: model 16800, trained on 2026-07-20 (see `note.txt`). This is the
  default policy used by `deploy_sf_tron1_arm_mujoco.py`.

Each runtime-ready ONNX set is located at `<policy-name>/
exported/` and contains
`actor.onnx`, `contactNet.onnx`, and `gru.onnx`.
