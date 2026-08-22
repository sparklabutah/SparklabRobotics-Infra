# relay

The FastAPI server between the Quest headset and everything else: a WebSocket
broadcast bus for pose/state messages, and a WebRTC publisher for camera video.
Robot-agnostic.

Console script: **`sparklab-relay`** (from `pip install -e ".[relay]"`).

```
server.py       the whole server — WS relay, WebRTC, camera readers
web/index.html  the page the Quest loads
web/client.js   WebXR session, pose sampling, camera quads, haptics
```

Message schemas and the Python subscriber side are in
[`../quest/README.md`](../quest/README.md).

## Running it

```bash
# LAN, real headset — WebXR requires a secure context, so TLS is mandatory
sparklab-relay --host 0.0.0.0 \
    --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem

# USB tether — localhost counts as secure by origin, no cert needed
sparklab-relay                      # binds 127.0.0.1:8443
adb reverse tcp:8443 tcp:8443       # then browse to http://localhost:8443
```

Generate the LAN cert with `scripts/make_certs.sh`. It costs a one-time
"advanced → proceed" click in the headset browser per certificate.

## Two responsibilities

**Broadcast relay.** Any message whose `type` is in `RELAY_TYPES` is forwarded
verbatim to every *other* connected client. Senders are excluded from their own
broadcast; unknown types are echoed to the sender only.

**WebRTC camera publisher.** On `webrtc_request` the server opens whatever
cameras exist, builds one `VideoStreamTrack` per camera, and exchanges SDP over
the same WebSocket. There is no second port. Trickle ICE from the browser is
dropped — aiortc gathers all local candidates before offering.

Camera specs come from `CAM_TOP`/`CAM_LEFT`/`CAM_RIGHT` env vars (legacy fixed
3-slot v4l2) or, if `VR_TELEOP_CAMERAS_YAML` is set, a YAML file listing
arbitrary cameras by id — v4l2 by device path or RealSense by serial. The
YAM-Ultra's rig file is `robots/yam_ultra/config/cameras.yaml`; camera ids
there become both the VR panel slots and the LeRobot dataset keys
(`observation.images.<id>`).

## Latency

Browse with `?latency=1` and the page measures round-trip time and posts a
`latency_report`, which the server logs. The `host` field self-labels the
transport, so USB-tether and LAN numbers compare directly.
