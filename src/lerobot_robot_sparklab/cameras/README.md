# cameras

RealSense setup tooling — operator utilities you run by hand, not part of any
control loop. The relay and `lerobot-record` open cameras themselves at runtime.

Per-rig camera *identity* (serials, ids, resolutions) lives with the robot, in
e.g. `robots/yam_ultra/config/cameras.yaml`.

```
calibrate_camera.py     find good exposure/white-balance/gain by eye
apply_camera_presets.py push a saved advanced-mode preset onto the hardware
export_intrinsics.py    dump every connected camera's intrinsics to YAML
streaming.py            legacy standalone MJPEG viewer — see below
```

RealSense autoexposure hunts when the camera is read at a fixed frame rate, as
the relay and dataset recording both do, so each camera's settings are frozen
once and stored per camera id in the rig's `cameras.yaml`.

## Two ways to freeze — do not mix them for one camera

**Advanced-mode preset** (what the YAM-Ultra rig uses). Export a preset from
`realsense-viewer`'s "Save settings to file", point the camera's
`advanced_json` field at it, and push it:

```bash
python -m lerobot_robot_sparklab.cameras.apply_camera_presets
```

Run once per power-up, before the relay or `lerobot-record`. The settings
persist in the device's own memory until it loses power.

**Inline three-setting alternative.** Set `exposure` / `white_balance` / `gain`
in `cameras.yaml` and the relay's reader applies them on every connect. Derive
the values with:

```bash
python -m lerobot_robot_sparklab.cameras.calibrate_camera --serial 323622272781
```

Tweak the flags, re-run, eyeball the feed in `realsense-viewer`, then paste
what looks right into the YAML.

Setting both for one camera makes the preset and the inline settings contend,
with the winner decided by ordering.

## Intrinsics

```bash
python -m lerobot_robot_sparklab.cameras.export_intrinsics
```

Walks every connected RealSense, starts each stream briefly, and writes
intrinsics to YAML. Needed for 3D reasoning from these frames; irrelevant for
policies consuming raw pixels.

## `streaming.py` is legacy

A standalone Flask MJPEG viewer, kept for checking that cameras are alive from
a browser. Not wired into the stack, and:

- serial numbers are hardcoded at the top of the file, not read from
  `cameras.yaml`, so they drift out of sync with the rig
- it opens all cameras at import time, so importing it has side effects
- Flask is not a declared dependency of this package

One process per RealSense serial, as everywhere else: running this while the
relay is up means one of them fails to open the device. For a live view during
teleop, use the relay's own WebRTC panels.
