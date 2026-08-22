# relay

The FastAPI server sitting between the Quest headset and everything else: a
WebSocket broadcast bus for pose/state messages, and a WebRTC publisher for
camera video. Robot-agnostic — it moves poses and frames, and does not know
what is listening.

Console script: **`sparklab-relay`** (from `pip install -e ".[relay]"`).

```
server.py       the whole server — WS relay, WebRTC, camera readers
web/index.html  the page the Quest loads
web/client.js   WebXR session, pose sampling, camera quads, haptics
```

For the message schemas and how to subscribe from Python, see
[`../quest/README.md`](../quest/README.md) — that's the consumer side of this
server, documented in one place rather than twice.

## Running it

```bash
# LAN, real headset — WebXR requires a secure context, so TLS is mandatory
sparklab-relay --host 0.0.0.0 \
    --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem

# USB tether — localhost counts as secure by origin, no cert needed
sparklab-relay                      # binds 127.0.0.1:8443
adb reverse tcp:8443 tcp:8443       # then browse to http://localhost:8443
```

The self-signed LAN cert costs you a one-time "advanced → proceed" click in
the headset browser per certificate.

## Two responsibilities

**1. Broadcast relay.** Any message whose `type` is in `RELAY_TYPES` is
forwarded verbatim to every *other* connected client. Senders are excluded
from their own broadcast. Unknown types are echoed to the sender only and go
nowhere else — so adding a new message type means adding it to that set.

**2. WebRTC camera publisher.** On `webrtc_request` the server opens whatever
cameras exist, builds one `VideoStreamTrack` per camera, and exchanges SDP
over the *same* WebSocket. There is no second port.

Camera specs come from either `CAM_TOP`/`CAM_LEFT`/`CAM_RIGHT` env vars (the
legacy fixed 3-slot v4l2 setup) or, if `VR_TELEOP_CAMERAS_YAML` is set, a YAML
file listing arbitrary cameras by id — v4l2 by device path or RealSense by
serial. The YAM-Ultra's rig file is
`robots/yam_ultra/config/cameras.yaml`; camera *ids* there become both the VR
panel slots and the LeRobot dataset keys (`observation.images.<id>`).

Trickle ICE from the browser is deliberately dropped: aiortc gathers all local
candidates before sending the offer, so on a LAN the connection establishes
from the SDP alone.

## Latency

Browse with `?latency=1` and the page measures round-trip time and posts a
`latency_report`, which the server logs. The `host` field self-labels the
transport, so USB-tether and LAN numbers can be compared directly off the
workstation terminal.
