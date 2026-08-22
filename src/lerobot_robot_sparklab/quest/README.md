# quest

Robot-agnostic Meta Quest input: controller and headset poses and buttons,
read off the relay's WebSocket. Turning those into joint targets belongs to a
robot's own teleoperator under `robots/<name>/teleop/`.

```
xr_client.py   XRFrameClient — background subscriber holding the newest frame
buttons.py     xr-standard button indices and accessors
```

## Topology

```
  Quest browser                    relay                      your code
  ┌──────────────┐                ┌────────────────┐      ┌──────────────┐
  │ WebXR page   │── xr_frame ───►│                │─────►│ XRFrameClient│
  │ (relay/web/) │◄── ik_state ───│  server.py     │◄─────│ .send(...)   │
  │              │◄══ WebRTC video ═══════════════ │      └──────────────┘
  └──────────────┘                └────────────────┘
```

The Quest runs a browser — no app, no SDK. It loads a page over HTTPS, that
page reads controllers via WebXR, and it pushes JSON over a WebSocket.

## Endpoints

Served by `relay/server.py` (`sparklab-relay`), default port **8443**:

| endpoint | purpose |
|---|---|
| `GET /` | the WebXR page the Quest loads |
| `GET /static/*` | page assets |
| `WS /ws` | all pose/state traffic — the one you subscribe to |

WebRTC camera streams are negotiated over that same socket, so there is no
second port. TLS is required for WebXR on a real headset; bind localhost and
tunnel over USB instead and `http://localhost:8443` counts as secure by origin.

## Message types

`server.py` rebroadcasts any message whose `type` is in `RELAY_TYPES` verbatim
to every *other* connected client. The sender is excluded from its own
broadcast. An unknown `type` is echoed to the sender only and relayed nowhere,
so adding a message type means adding it to that set.

| type | direction | payload |
|---|---|---|
| `xr_frame` | Quest → subscribers | controller + headset pose, buttons, axes |
| `ik_state` | teleop → Quest | joint angles, engage state, haptic intensity |
| `config_update` | either | runtime setting changes |
| `request_settings` | either | ask the peer to re-send its settings |
| `haptic_calibrate` / `_result` | web UI ↔ teleop | gripper-haptic threshold calibration |

On connect the server also sends, unsolicited:

```json
{"type": "camera_list", "cameras": [{"id": "top", "label": "..."}]}
```

## `xr_frame` schema

Sent once per headset display frame — 72–120 Hz, far faster than any control
loop consuming it.

```json
{
  "type": "xr_frame",
  "t_client": 12345.6,
  "controllers": {
    "left":  {
      "position":    [x, y, z],
      "orientation": [x, y, z, w],
      "buttons": [{"p": false, "t": false, "v": 0.0}, ...],
      "axes":    [0.0, 0.0, ...]
    },
    "right": { ... }
  },
  "viewer": { "position": [x, y, z], "orientation": [x, y, z, w] }
}
```

- `t_client` — browser `performance.now()` in ms. Not wall-clock, and shares
  no epoch with the host.
- `buttons[i]` — `p` pressed, `t` touched, `v` analog 0..1. Every button the
  controller reports, uncurated.
- `axes` — every axis, raw. Thumbstick is typically `axes[2]`/`axes[3]`.
- `viewer` — headset pose, or `null` if unavailable this frame.
- Only `left`/`right` handedness is included. If no controller is tracked no
  frame is sent at all, so silence means "nothing tracked", not "nothing moved".

`position` is **wrist-shifted, not raw**: shifted along the controller's local
frame to the operator's wrist pivot, so a pure wrist twist produces ≈zero
translation. The offset comes from the in-VR calibration flow. Only the
calibration buffers keep unshifted samples.

### Coordinate frame

WebXR `local-floor`: right-handed, **Y up**, **−Z forward** (whichever way the
operator faced at session start), X right, origin on the floor. Quaternions are
`[x, y, z, w]` — scalar **last**, the opposite of MuJoCo's `[w, x, y, z]`.

The page falls back to `local` where `local-floor` is unsupported, which puts
the origin at head height. Clutch-relative consumers are unaffected since they
use pose deltas only.

## Reading inputs

```python
from lerobot_robot_sparklab.quest import XRFrameClient, buttons

client = XRFrameClient("wss://127.0.0.1:8443/ws")
client.connect(timeout_s=5.0)

frame, age_s = client.latest()
if frame is not None:
    ctrl = (frame.get("controllers") or {}).get("right")
    if ctrl:
        pos = ctrl["position"]
        clutching = buttons.pressed(ctrl["buttons"], buttons.GRIP)
        grip_amount = buttons.value(ctrl["buttons"], buttons.TRIGGER)
```

Publish back with `send()`, which is fire-and-forget and safe from a non-async
thread:

```python
client.send({"type": "ik_state", "right_qpos": [...], "right_haptic": 0.4})
```

### Button indices

| constant | index | typical use |
|---|---|---|
| `TRIGGER` | 0 | analog — gripper closure |
| `GRIP` | 1 | clutch |
| `THUMBSTICK` | 3 | thumbstick press |
| `A_X` | 4 | A (right) / X (left) |
| `B_Y` | 5 | B (right) / Y (left) |

A controller reporting a shorter array reads as "not pressed" forever past its
end, silently. Log `buttons.describe_layout(ctrl["buttons"], buttons.A_X,
buttons.B_Y)` once per hand at startup so that names itself.

### Staleness

`latest()` returns the frame's **age**, not a verdict. Gate *motion* on
`age_s < buttons.XR_FRAME_STALE_TIMEOUT_S` (0.2 s) and let safety actions —
stow, disarm — through regardless. `XRFrameClient` reconnects on its own, so a
dropout raises age rather than ending anything.

## Behaviour worth knowing

- **In-VR calibration suppresses `xr_frame` entirely.** While the operator
  captures the wrist-pivot offset, the page stops streaming. A subscriber sees
  a multi-second silence, not an error.
- **Self-signed TLS is accepted** for `wss://`. The headset browser also wants
  a one-time "advanced → proceed" per certificate.
- **One writer per topic.** Two processes publishing `ik_state` to one relay
  fight, and neither observes it, since senders are excluded from their own
  broadcast.
