# Quest side

How controller and headset input gets from a Meta Quest into Python, and how
to subscribe to it from new code without touching the relay or any existing
consumer.

Nothing here is robot-specific. `quest/` gives you poses and buttons; turning
those into joint targets belongs to a robot's own teleoperator under
`robots/<name>/teleop/`.

## Topology

```
  Quest browser                    relay (this repo)              your code
  ┌──────────────┐                ┌────────────────┐          ┌──────────────┐
  │ WebXR page   │── xr_frame ───►│                │──────────►│ XRFrameClient│
  │ (web/        │                │  server.py     │  (fan-out │              │
  │  client.js)  │◄── ik_state ───│  broadcast     │◄──────────│ .send(...)   │
  │              │                │                │           └──────────────┘
  │              │◄══ WebRTC video ═══════════════ │
  └──────────────┘                └────────────────┘
```

The Quest runs a **browser**. There is no app to install and no SDK — the
headset loads a page over HTTPS, that page uses the WebXR API to read
controllers, and it pushes JSON over a WebSocket. Everything downstream is a
subscriber to that one socket.

## Endpoints

Served by `relay/server.py` (console script `sparklab-relay`), default port
**8443**:

| endpoint | purpose |
|---|---|
| `GET /` | the WebXR page the Quest loads |
| `GET /static/*` | page assets (`client.js` and friends) |
| `WS /ws` | **the one you subscribe to** — all pose/state traffic |

WebRTC camera streams are negotiated over that same `/ws` socket (SDP offer
and answer as WebSocket messages), so there is no second port to open.

TLS is required for WebXR on a real headset — browsers gate the API behind a
secure context. Either serve HTTPS with a self-signed LAN cert:

```bash
sparklab-relay --host 0.0.0.0 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

…or bind localhost and tunnel over USB, where `http://localhost:8443` counts
as secure by origin.

## Message types

`server.py` rebroadcasts any message whose `type` is in `RELAY_TYPES`
**verbatim to every other connected client**:

| type | direction | payload |
|---|---|---|
| `xr_frame` | Quest → subscribers | controller + headset pose, all buttons, all axes |
| `ik_state` | teleop → Quest | joint angles, engage state, haptic intensity |
| `config_update` | either | runtime setting changes |
| `request_settings` | either | ask the peer to re-send its settings |
| `haptic_calibrate` / `_result` | web UI ↔ teleop | gripper-haptic threshold calibration |

Plus one server-generated message you get **unsolicited on connect**, before
anything else:

```json
{"type": "camera_list", "cameras": [{"id": "top", "label": "..."}]}
```

Two properties worth knowing before you design around this:

- **The sender is excluded from its own broadcast** (`_broadcast(raw,
  exclude=websocket)`). Two teleop processes on one relay hear each other but
  not themselves.
- **`RELAY_TYPES` is an allowlist.** An unknown `type` is echoed back to the
  sender only — it is *not* relayed. Adding a new message type is a one-line
  edit to that set in `server.py`. This is the one place "just subscribe"
  isn't literally true.

## `xr_frame` schema

Sent from `client.js` once per headset display frame — 72–120 Hz depending on
device and settings, i.e. far faster than any control loop that consumes it.

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

- `t_client` — browser `performance.now()` in ms. Useful for latency
  measurement; it is **not** wall-clock and shares no epoch with the host.
- `buttons[i]` — `p` pressed (bool), `t` touched (bool), `v` analog 0..1.
  Every button the controller reports, uncurated.
- `axes` — every axis, raw. Thumbstick is typically `axes[2]`/`axes[3]`.
- `viewer` — headset pose, or `null` if the pose was unavailable this frame.
  Used to yaw-correct the engage frame to wherever the operator is facing.
- Only `left`/`right` handedness is included; a hand-tracking or unlabelled
  input source is skipped. If no controller is tracked, no frame is sent at
  all — so silence means "nothing tracked", not "nothing moved".

### Coordinate frame

WebXR `local-floor`: right-handed, **Y up**, **−Z forward** (the direction the
operator faced at session start), X right, origin on the floor. Quaternions
are `[x, y, z, w]` — scalar **last**, which is the opposite of MuJoCo's
`[w, x, y, z]`.

The page falls back to `local` if `local-floor` is unsupported, and that
fallback puts the origin at head height instead of the floor. Y is then offset
by roughly the operator's standing height. Anything that cares about absolute
height should not assume the floor datum; clutch-relative consumers (the usual
case) are unaffected because they only ever use pose *deltas*.

### Poses are wrist-shifted, not raw

`position` is not the controller's grip origin. It is shifted along the
controller's local frame to the operator's wrist pivot, so a pure wrist twist
produces ≈zero translation. The offset comes from the in-VR calibration flow.
This is what you want for teleoperation; if you need the true grip pose, note
that only the *calibration* buffers keep the unshifted samples.

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

Publish back to the headset — anything in `RELAY_TYPES` — with `send()`:

```python
client.send({"type": "ik_state", "right_qpos": [...], "right_haptic": 0.4})
```

`send()` is fire-and-forget and safe to call from a non-async thread; it
hands the coroutine to the client's own event loop.

### Button indices

xr-standard layout, in `buttons.py`:

| constant | index | typical use |
|---|---|---|
| `TRIGGER` | 0 | analog — gripper closure |
| `GRIP` | 1 | clutch |
| `THUMBSTICK` | 3 | thumbstick press |
| `A_X` | 4 | A (right) / X (left) |
| `B_Y` | 5 | B (right) / Y (left) |

A controller that reports a **shorter** button array reads as "not pressed"
forever at every index past its end, silently. That has cost us real debugging
time — a stow button that simply never fired, with nothing in the log. Log
`buttons.describe_layout(ctrl["buttons"], buttons.A_X, buttons.B_Y)` once per
hand at startup and the failure names itself.

### Staleness

`latest()` returns the frame's **age**, not a freshness verdict, and the
caller decides per action. Do not reflexively drop stale frames: a stow or
disarm button is exactly what an operator reaches for *during* a bad
connection, and an early return eats the press. Gate *motion* on
`age_s < buttons.XR_FRAME_STALE_TIMEOUT_S` (0.2 s); let safety actions
through regardless.

`XRFrameClient` reconnects on its own — WiFi and USB-tunnel jitter drop the
socket routinely mid-session — so a dropout raises age rather than ending
anything.

## Gotchas

- **In-VR calibration suppresses `xr_frame` entirely.** While the operator is
  capturing the wrist-pivot offset (both grips squeezed in calibration mode),
  the page stops streaming so downstream teleop doesn't chase calibration
  motion. A subscriber sees this as a multi-second silence, not an error.
- **Self-signed TLS.** `XRFrameClient` disables verification for `wss://` on
  purpose — the LAN cert is self-signed and validation would only refuse the
  handshake. The headset browser will also demand a one-time "advanced →
  proceed" click per cert.
- **Don't add a second writer to a topic.** Two processes publishing
  `ik_state` to one relay will fight, and because senders are excluded from
  their own broadcast neither will observe the conflict.

## Extending

To consume Quest input from something new — a logger, a different robot, a
dataset annotator — construct an `XRFrameClient` and read. No relay change,
no coordination with existing consumers, and the relay fans out to all of
them.

Adding a *new message type* is the one case needing a relay edit: add it to
`RELAY_TYPES` in `server.py`, or it will be echoed to the sender and dropped.

ROS 2 is deliberately not in this path. The Quest is a browser and cannot
speak DDS, so ROS 2 would sit *behind* a WebSocket bridge rather than replace
one. If a natively-ROS 2 robot joins the lab, the clean move is a bridge node
that subscribes here and republishes as topics — purely additive, and it keeps
the ROS 2 dependency scoped to the robot that needs it.
