# sparklab_cameras

The RealSense cameras on ROS 2, plus tools to set them up. Which cameras exist is
set in [`sparklab_yam/config/cameras.yaml`](../sparklab_yam/config/README.md).

```mermaid
flowchart LR
    rs([3 RealSense cameras]) -->|USB| cams[cameras]
    cams -->|"/cameras/*/color/image_raw<br/>/cameras/*/depth/image_raw<br/>/cameras/*/*/camera_info<br/>/cameras/*/points"| users["quest_bridge, LeRobot,<br/>agents, Foxglove"]
```

```
sparklab_cameras/node.py               the camera node
sparklab_cameras/apply_presets.py      load saved settings onto the cameras
sparklab_cameras/calibrate.py          try exposure / white balance / gain values
sparklab_cameras/export_intrinsics.py  save every camera's intrinsics to YAML
```

## The camera node

`yam.launch.py` starts it. It is the only thing that opens the cameras; everything
else subscribes. A camera that can't be opened is skipped with an error.

| topic | type | |
|---|---|---|
| `/cameras/<id>/color/image_raw` | `Image` `rgb8` | colour |
| `/cameras/<id>/depth/image_raw` | `Image` `16UC1` | depth in **mm**, 0 = no reading |
| `/cameras/<id>/{color,depth}/camera_info` | `CameraInfo` | intrinsics |
| `/cameras/<id>/points` | `PointCloud2` | xyz (m) + rgb, ≤ 10 Hz, only while something subscribes |

Depth is aligned to colour: pixel (u, v) is the same point in both, so they share
intrinsics. `frame_id` is the camera id.

## Camera settings

Settings are fixed so auto-exposure doesn't flicker in recordings. Per camera, either:

- **a saved preset** (what the rig uses): save it from `realsense-viewer`, point the
  camera's `advanced_json` at it, and load it after each power-up:

  ```bash
  ros2 run sparklab_cameras apply_presets src/sparklab_yam/config/cameras.yaml
  ```

- **three values** in `cameras.yaml` (`exposure`, `white_balance`, `gain`), applied
  on every start. Find them with `ros2 run sparklab_cameras calibrate --serial <serial>`.

## Intrinsics

```bash
ros2 run sparklab_cameras export_intrinsics intrinsics.yaml
```
