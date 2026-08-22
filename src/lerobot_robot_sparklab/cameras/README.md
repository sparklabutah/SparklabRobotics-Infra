# cameras

RealSense setup tooling. These are **operator utilities you run by hand**, not
part of any control loop — the relay and `lerobot-record` open cameras
themselves at runtime.

Per-rig camera *identity* (serials, ids, resolutions) is not here. It lives
with the robot, e.g. `robots/yam_ultra/config/cameras.yaml`, because which
cameras exist is a property of a rig and not of RealSense.

```
calibrate_camera.py     find good exposure/white-balance/gain by eye
apply_camera_presets.py push a saved advanced-mode preset onto the hardware
export_intrinsics.py    dump every connected camera's intrinsics to YAML
streaming.py            legacy standalone MJPEG viewer — see the warning below
```

## Why cameras need freezing at all

RealSense autoexposure *hunts* when the camera is read at a fixed frame rate,
which is exactly what the relay and dataset recording both do. The picture
visibly pulses. Worse, for a recorded dataset it means identical scenes get
different pixel statistics depending on when they were captured — the policy
then has to learn around an artifact of the camera's control loop.

So each camera gets its settings frozen once, and the values are stored per
camera id in the rig's `cameras.yaml`.

## Two ways to freeze, don't mix them

**Advanced-mode preset (what the YAM-Ultra rig uses).** Export a full preset
from `realsense-viewer`'s "Save settings to file", point the camera's
`advanced_json` field at it, and push it:

```bash
python -m lerobot_robot_sparklab.cameras.apply_camera_presets
```

Run **once per power-up**, before the relay or `lerobot-record`. The settings
persist in the *device's own memory* until it loses power, so later runs just
see an already-locked camera and reapply nothing.

**Inline three-setting alternative.** Set `exposure` / `white_balance` /
`gain` in `cameras.yaml` and the relay's reader applies them on every connect
— no separate script, no advanced mode. Derive the values with:

```bash
python -m lerobot_robot_sparklab.cameras.calibrate_camera --serial 323622272781
```

Tweak the flags, re-run, eyeball the feed in `realsense-viewer`, then paste
what looks right into the YAML.

**Don't set both for the same camera** — the preset and the inline settings
will contend, and which wins depends on ordering rather than intent.

## Intrinsics

```bash
python -m lerobot_robot_sparklab.cameras.export_intrinsics
```

Walks every connected RealSense, starts each stream briefly, and writes
intrinsics to YAML. Needed for anything doing 3D reasoning from these frames;
irrelevant for policies that consume raw pixels.

## `streaming.py` is legacy — read before using

A standalone Flask MJPEG viewer, kept because it's occasionally handy for
checking that three cameras are alive from a browser. It is **not** wired into
the stack, and it has three sharp edges:

- **Serial numbers are hardcoded** at the top of the file with a `TODO`, and
  they are not read from `cameras.yaml`. They will drift out of sync with the
  rig.
- **It opens all cameras at import time**, so importing it has side effects.
- **Flask is not a declared dependency** of this package — you must install it
  yourself.

And, as everywhere else: one process per RealSense serial. Running this while
the relay is up means one of them fails to open the device. For a live view
during teleop, use the relay's own WebRTC panels instead.
