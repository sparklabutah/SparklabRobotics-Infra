# sparklab_sim

Isaac Sim digital twin of the lab's rigs. **Kinematic, not dynamic**: it poses
arms and renders cameras; it does not simulate contact, torque or grasping.

```
paths.py         locate shared assets in the sibling package, by path
rig.py           measured table layout — what is MEASURED vs ASSUMED
convert.py       URDF -> USD, run once
scene.py         build the scene, pose arms, place/move cameras
render.py        camera -> ndarray, side-by-side, diff metric
rigcams.py       the rig's camera set, matched to the real intrinsics
capture.py       on-demand capture server
viewport.py      interactive viewport: MJPEG out, mouse in (used by live.py)
live.py          edit the .usda, orbit it in a browser, over SSH
policy_server.py HTTP server the `yam_ultra_sim` robot talks to
```

Everything here runs under Isaac's bundled Python, which has no `lerobot` —
that is the only reason this package is separate, and it makes the boundary a
mechanical test: *a module that imports lerobot does not belong here.* Shared
assets are read by path out of the sibling package (`paths.py`). The rollout
runner lives on the other side of that line, in
[`../lerobot_robot_sparklab/rollout/`](../lerobot_robot_sparklab/rollout/). See
[`DESIGN.md`](../../DESIGN.md#package-boundaries).

## Running

There is no conda env to activate, and you should not be in one. The wrapper
drops any active env and puts `<repo>/src` on `PYTHONPATH`:

```bash
./scripts/isaac_python.sh -m sparklab_sim.convert        # URDF -> USD, once
./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics
./scripts/isaac_python.sh my_script.py
./scripts/isaac_python.sh                                # REPL
```

Safe to call from inside `robot-py312` — it deactivates first. Point at another
install with `ISAAC=/path/to/...`.

`--physics` is required for anything involving **contact**. Without it the
fingers teleport and cannot hold an object, so every grasp fails regardless of
the policy.

## Editing the scene over SSH

```bash
./scripts/isaac_python.sh -m sparklab_sim.live      # -> http://127.0.0.1:8080/
```

Drag to orbit, wheel to zoom, shift-drag to pan. Edit `sim/scene.usda` in any
editor and save; the view follows in ~0.3 s. `.usda` is ASCII, so moving a prim
is editing its `xformOp:translate`. The camera dropdown switches between free
look and the scene's authored cameras. `copy pose` puts the current view on the
clipboard as a `set_orbit_camera(...)` call, which is how a view you liked
becomes a number in `rig.py`.

Flags: `--play` streams continuously, `--res` sets free-look resolution,
`--once --out x.png` takes a headless snapshot, `--reopen` forces full stage
reopens, `--regenerate` rebuilds the scene from `rig.py` **discarding your
edits**. `/frame.jpg` is a single still for scripting.

In VS Code over SSH the port forwards itself — check the **Ports** panel.
`--host 0.0.0.0` serves to the network instead.

**Call `Renderer.close()` before reopening a stage**, or the next update
segfaults. The sublayer reload scheme that makes an edit cost ~0.04 s is
described in [`DESIGN.md`](../../DESIGN.md#the-live-viewers-fast-path).

## Live view from a notebook

For tweaking driven by Python rather than by the file:

```python
from sparklab_sim import render, viewport

r  = render.Renderer(app, scene.CAMERA_PRIM)     # build ONCE
vp = viewport.Viewport(port=8890); vp.start()    # -> http://127.0.0.1:8890/

scene.set_camera(stage, standoff=1.3, height=0.9, pitch_deg=-25)
vp.publish(r.frame(settle=4))                    # tab updates
```

Reuse the `Renderer` and `set_camera`; do not rebuild the stage. Isaac costs
~15 s to boot and holds the GPU, so a warm kernel mutating the camera iterates
in seconds. Call `r.close()` before opening another stage.

`render.abs_diff` returns mean absolute error in 0-255. Two renders of an
unchanged scene differ by **MAE ≈ 0.8** from RTX temporal sampling — that is
the noise floor, so a camera fit scoring below it is done. Raise `settle` for a
quieter image.

## Notebooks in VS Code over SSH

Isaac's own kernelspec does not work in VS Code, which launches kernels itself
and so fails with `ModuleNotFoundError: No module named 'isaacsim'`. Install
the self-contained kernel, which bakes the resolved environment and this repo's
`src/` into `kernel.json`:

```bash
./scripts/install_isaac_kernel.sh          # re-run if Isaac or the repo moves
```

Then, in the **remote** window (extensions are per-host): install the Python and
Jupyter extensions, open a notebook, and pick **Select Kernel → Jupyter
Kernel... → SparkLab Isaac Sim (3.12)**. Not **Python Environments...**, which
lists conda interpreters, none of which can run Isaac.

If the kernel refuses to appear, run `./scripts/start_isaac_jupyter.sh` and use
**Select Kernel → Existing Jupyter Server...** with the printed tokenised URL.

No X server is needed — the app runs `headless=True` and frames come back
through Replicator as arrays. First launch is ~15 s warm, several minutes the
very first time while RTX shaders compile. The kernel holds the GPU for as long
as it is alive, so interrupt it before starting another Isaac process.
`app.close()` kills the kernel; that is expected, not a crash.

## Constraints to know

- **`SimulationApp` must be constructed before any `omni.*` / `isaacsim.*`
  import.** Those modules do not exist until Kit has booted, which is why every
  Isaac import here sits inside a function, never at module scope.
- **Placing an imported arm needs USD xform ops on the reference prim**, not
  `Articulation.set_world_poses`, and it must happen before the timeline plays:
  the asset is imported `fix_base=True`, so on the first physics step each arm
  welds to the world wherever it stands. Get this wrong and both arms silently
  stack at the origin.
- **Joint writes need the timeline playing.** `set_dof_positions` asserts
  "physics tensor entity is not valid" until then; `scene.start()` plays it.
- **Gravity is off on purpose.** The URDF ships no joint drive gains, so under
  gravity the joints sag out of any pose written to them.
- **Link paths carry a `Geometry` scope** — `<arm>/Geometry/base/link1/...`.
  Use `scene.link_path()`; it has changed once already.

## The USD is a build artifact

`convert.py` writes `robots/yam_ultra/model/usd/`, which is gitignored. It
derives from the vendored `yam_ultra.urdf`, not i2rt's installed copy.

Verified after conversion: all six joint limits survive into the USD (stored in
degrees — `joint3` reads `171.88734°` = 3.0 rad), and forward kinematics agree
with the MuJoCo model to **0.4 mm**.

## What is still missing

`rig.py` marks every value MEASURED or ASSUMED. Measured: arm separation
(0.60 m centre-to-centre) and that the third-person camera is laterally
centred. Assumed: arm yaw, camera standoff/height/pitch, table size.

**Wrist camera extrinsics are absent entirely** — the URDF has no camera links
and no hand-eye calibration has been done. That calibration is the prerequisite
for comparing rendered frames against recorded ones.
