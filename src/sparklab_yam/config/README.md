# config

Settings for *this* physical rig.

```
yam.yaml                  ROS parameters: drivers, teleop, calibration board
cameras.yaml              which cameras exist: id, serial, resolution
extrinsics.yaml           calibrated camera and arm-base poses, published as static TF
hand_default.json         saved RealSense settings for the wrist cameras
thirdperson_default.json  saved RealSense settings for the top camera
intrinsics.yaml           camera intrinsics
```

## cameras.yaml

| id | serial | model | capture |
|---|---|---|---|
| `left_wrist` | 353322271147 | D405 | 640×480 @ 30 |
| `right_wrist` | 323622272781 | D405 | 640×480 @ 30 |
| `top` | 243322071190 | D435F | 1280×720 @ 15 |

**The ids are names everywhere:** topic names (`/cameras/<id>/color/...`) and
dataset columns (`observation.images.<id>`). Renaming an id renames a dataset column.

Per-camera keys: `serial`, `width`, `height`, `fps`, `crop_height` (rows cut off
top and bottom), and either `advanced_json` (a saved settings file) or
`exposure` / `white_balance` / `gain`.

## Replacing a camera

Update its serial in `cameras.yaml`, load its settings again (`apply_presets`),
and re-export the intrinsics. A wrong serial shows up as a camera that won't
open, or as the two wrist cameras silently swapped.
