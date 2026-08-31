"""FastAPI relay + WebRTC publisher for VR teleop.

Per WebSocket client, it does two things: broadcasts pose/state messages
(RELAY_TYPES below) between the Quest browser and the teleop process, and
publishes camera tracks over WebRTC. On webrtc_request it opens whichever
configured cameras exist, builds an RTCPeerConnection with one track each, and
exchanges SDP/ICE over the same socket. `camera_toggle` mutes a track by
repeating its last frame, which costs near-zero bandwidth under H.264 and
needs no renegotiation.

    Quest browser  ── xr_frame ─────►  server  ── xr_frame ──►  teleop process
                   ◄── ik_state ──     server  ◄── ik_state ──
                   ◄═ WebRTC video ═   server

Run::

    sparklab-relay                          # 127.0.0.1, for USB / tunnel
    sparklab-relay --host 0.0.0.0 \
        --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem   # LAN HTTPS
"""

import argparse
import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vr_teleop")
WEB_DIR = Path(__file__).parent / "web"

# WebSocket message types broadcast verbatim to every other connected client.
# Anything not in this set is handled inline (signaling, toggles, ping).
RELAY_TYPES = {
    "xr_frame", "ik_state", "config_update", "request_settings",
    # Web UI ↔ teleop: gripper-haptic threshold calibration.
    "haptic_calibrate", "haptic_calibrate_result",
}


# ── Camera capture ───────────────────────────────────────────────────────
# One reader per device: a blocking cv2 loop feeding a lock-protected slot.

@dataclass(frozen=True)
class CameraSpec:
    """Camera identity + capture parameters.

    id: matches the WS schema, by convention top | left_wrist | right_wrist.
    label: human-facing name.
    rotate: 0/90/180/270, applied in the capture thread so every consumer sees
        the corrected frame.
    backend: "v4l2" (`device` is a /dev/videoN path) or "realsense" (`serial`
        selects the camera; exposure/white_balance/gain freeze the sensor).
    crop_height: if smaller than `height`, centre-crops that many rows to match
        a target video shape the sensor cannot capture natively. Applied after
        `rotate`, so it always crops the final on-screen height.
    """
    id: str
    label: str
    device: str | None
    width: int
    height: int
    fps: int
    rotate: int  # 0 | 90 | 180 | 270
    backend: str = "v4l2"  # "v4l2" | "realsense"
    serial: str | None = None
    exposure: int | None = None
    white_balance: int | None = None
    gain: int | None = None
    crop_height: int | None = None


# cv2 rotation lookup. 0 → no rotation; other values map to the cv2 constants.
_ROTATE_CODES = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _center_crop_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """Crop rows evenly off the top and bottom to reach `target_h`. No-op if
    `target_h` isn't smaller than the frame already is."""
    h = frame.shape[0]
    if target_h is None or target_h >= h:
        return frame
    top = (h - target_h) // 2
    return frame[top: top + target_h]


class CameraReader:
    """Background v4l2 grabber. Thread-safe latest-frame slot."""

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._cap = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {spec.id} at {spec.device}")
        # MJPG is much cheaper than YUYV for these UVC cams at 640x480@30.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
        self._cap.set(cv2.CAP_PROP_FPS, spec.fps)
        self._thread = threading.Thread(
            target=self._loop, name=f"cam-{spec.id}", daemon=True
        )
        self._thread.start()
        logger.info("camera %s opened (%s, %dx%d@%d rotate=%d)",
                    spec.id, spec.device, spec.width, spec.height, spec.fps, spec.rotate)

    def _loop(self) -> None:
        rotate_code = _ROTATE_CODES.get(self.spec.rotate)
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            if self.spec.crop_height is not None:
                frame = _center_crop_height(frame, self.spec.crop_height)
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


class RealSenseCameraReader:
    """Background pyrealsense2 grabber, opened by serial.

    Same latest-frame slot interface as CameraReader, and freezes the sensor
    settings from the spec if given. pyrealsense2 is imported lazily, so
    v4l2-only setups never need it installed.
    """

    def __init__(self, spec: CameraSpec) -> None:
        import pyrealsense2 as rs

        self.spec = spec
        self._rs = rs
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(spec.serial)
        cfg.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.bgr8, spec.fps)
        profile = self._pipe.start(cfg)

        if spec.exposure is not None or spec.white_balance is not None or spec.gain is not None:
            sensor = profile.get_device().query_sensors()[0]  # D405/D415: one sensor for color
            if spec.exposure is not None:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, spec.exposure)
            if spec.white_balance is not None:
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
                sensor.set_option(rs.option.white_balance, spec.white_balance)
            if spec.gain is not None:
                sensor.set_option(rs.option.gain, spec.gain)

        self._thread = threading.Thread(
            target=self._loop, name=f"cam-{spec.id}", daemon=True
        )
        self._thread.start()
        logger.info("camera %s opened (realsense %s, %dx%d@%d rotate=%d)",
                    spec.id, spec.serial, spec.width, spec.height, spec.fps, spec.rotate)

    def _loop(self) -> None:
        rotate_code = _ROTATE_CODES.get(self.spec.rotate)
        last_frame_at = time.time()
        last_stall_warn = 0.0
        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                now = time.time()
                # No frame has ever arrived, or delivery stalled. Rate-limited
                # so a real outage doesn't spam every wait_for_frames retry.
                if now - last_frame_at > 3.0 and now - last_stall_warn > 5.0:
                    last_stall_warn = now
                    logger.warning("camera %s: no frame in %.1fs (wait_for_frames timing out) — "
                                    "check USB bandwidth/power or another process holding the device",
                                    self.spec.id, now - last_frame_at)
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())  # already BGR (rs.format.bgr8)
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            if self.spec.crop_height is not None:
                frame = _center_crop_height(frame, self.spec.crop_height)
            last_frame_at = time.time()
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._pipe.stop()


class CameraTrack(VideoStreamTrack):
    """aiortc track that pulls from a CameraReader. `enabled=False` returns
    the previously-sent frame, so H.264's P-frames compress to nothing and
    the receiver freezes on the last image — no SDP renegotiation needed."""

    kind = "video"

    def __init__(self, reader: CameraReader) -> None:
        super().__init__()
        self.reader = reader
        self.enabled = True
        self._last_sent: np.ndarray | None = None
        # Black fallback, so the encoder has something before the first frame.
        # Rotate then crop, matching _loop(), or the shapes disagree.
        h, w = reader.spec.height, reader.spec.width
        if reader.spec.rotate in (90, 270):
            h, w = w, h
        if reader.spec.crop_height is not None:
            h = min(h, reader.spec.crop_height)
        self._black = np.zeros((h, w, 3), dtype=np.uint8)

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        if self.enabled:
            frame = self.reader.latest()
            if frame is None:
                frame = self._last_sent if self._last_sent is not None else self._black
            else:
                self._last_sent = frame
        else:
            frame = self._last_sent if self._last_sent is not None else self._black
        # cv2 returns BGR; PyAV's bgr24 format matches.
        vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf


# ── Camera registry ──────────────────────────────────────────────────────
# Only cameras actually present are registered, so a missing one disappears.

def _load_camera_specs_from_yaml(path: Path) -> list[CameraSpec]:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    specs: list[CameraSpec] = []
    for cam_id, cfg in (data.get("cameras") or {}).items():
        model = str(cfg.get("model", ""))
        backend = "realsense" if model.upper().startswith("D4") else "v4l2"
        specs.append(CameraSpec(
            id=cam_id,
            label=cfg.get("label", cam_id.replace("_", " ").title()),
            device=cfg.get("device"),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
            rotate=cfg.get("rotate", 0),
            backend=backend,
            serial=str(cfg["serial"]) if "serial" in cfg else None,
            exposure=cfg.get("exposure"),
            white_balance=cfg.get("white_balance"),
            gain=cfg.get("gain"),
            crop_height=cfg.get("crop_height"),
        ))
    return specs


CAMERA_SPECS: list[CameraSpec] = []
CAMERA_READERS: dict[str, CameraReader | RealSenseCameraReader] = {}


def _ensure_readers() -> None:
    """Lazy-open the cameras on first WebRTC request. Avoids holding v4l2/
    RealSense locks during dev iterations where the operator only wants
    the relay."""
    for spec in CAMERA_SPECS:
        if spec.id in CAMERA_READERS:
            continue
        try:
            reader_cls = RealSenseCameraReader if spec.backend == "realsense" else CameraReader
            CAMERA_READERS[spec.id] = reader_cls(spec)
        except Exception:
            logger.exception("failed to open camera %s", spec.id)


# ── Codec preference ─────────────────────────────────────────────────────
# Quest decodes H.264 in hardware; VP8 is software-decoded, so worse latency.

def _prefer_h264(pc: RTCPeerConnection) -> None:
    caps = RTCRtpSender.getCapabilities("video")
    h264 = [c for c in caps.codecs if c.mimeType == "video/H264"]
    if not h264:
        return
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(h264)


# ── Per-WebSocket client state ───────────────────────────────────────────
# Each WS owns its peer connection and tracks, keyed by camera id.

@dataclass
class ClientState:
    ws: WebSocket
    pc: RTCPeerConnection | None = None
    tracks: dict[str, CameraTrack] | None = None  # id -> track


_clients: dict[WebSocket, ClientState] = {}


async def _broadcast(text: str, exclude: WebSocket | None = None) -> None:
    if not _clients:
        return
    targets = [ws for ws in list(_clients) if ws is not exclude]
    if not targets:
        return
    await asyncio.gather(
        *[ws.send_text(text) for ws in targets],
        return_exceptions=True,
    )


# ── FastAPI app ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for reader in CAMERA_READERS.values():
        reader.stop()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Force-disable browser cache for the page and JS so operators don't
    have to clear the Quest's cache after every web-asset change."""
    response = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


# ── WebRTC signaling ─────────────────────────────────────────────────────
# Server-as-publisher. Trickled ICE is dropped; the offer SDP carries ours.

async def _handle_webrtc_request(state: ClientState, msg: dict) -> None:
    if state.pc is not None:
        # Already negotiated for this WS — tear down before re-offering.
        await _close_pc(state)

    _ensure_readers()
    enabled_set: set[str] = set(msg.get("enabled_cameras") or [s.id for s in CAMERA_SPECS])

    pc = RTCPeerConnection()
    state.pc = pc
    state.tracks = {}

    for spec in CAMERA_SPECS:
        reader = CAMERA_READERS.get(spec.id)
        if reader is None:
            continue
        track = CameraTrack(reader)
        track.enabled = spec.id in enabled_set
        sender = pc.addTrack(track)
        # aiortc otherwise assigns a random UUID, leaving the client unable to
        # match a track to a UI slot. Private attr, stable across aiortc 1.9+.
        sender._stream_id = spec.id
        state.tracks[spec.id] = track

    _prefer_h264(pc)

    @pc.on("iceconnectionstatechange")
    async def _on_ice_state() -> None:
        logger.info("ice state: %s", pc.iceConnectionState)
        if pc.iceConnectionState in ("failed", "closed"):
            await _close_pc(state)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    await state.ws.send_text(json.dumps({
        "type": "webrtc_offer",
        "sdp": pc.localDescription.sdp,
        "sdp_type": pc.localDescription.type,
        "cameras": [
            {"id": spec.id, "label": spec.label}
            for spec in CAMERA_SPECS if spec.id in state.tracks
        ],
    }))


async def _handle_webrtc_answer(state: ClientState, msg: dict) -> None:
    if state.pc is None:
        logger.warning("webrtc_answer with no pending pc")
        return
    answer = RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"])
    await state.pc.setRemoteDescription(answer)


async def _handle_camera_toggle(state: ClientState, msg: dict) -> None:
    cam_id = msg.get("camera_id")
    enabled = bool(msg.get("enabled", True))
    if state.tracks and cam_id in state.tracks:
        state.tracks[cam_id].enabled = enabled
        logger.info("camera %s -> %s", cam_id, "on" if enabled else "off")


async def _close_pc(state: ClientState) -> None:
    if state.pc is None:
        return
    try:
        await state.pc.close()
    except Exception:
        pass
    state.pc = None
    state.tracks = None


# ── WebSocket endpoint ───────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_handler(websocket: WebSocket) -> None:
    await websocket.accept()
    state = ClientState(ws=websocket)
    _clients[websocket] = state
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    logger.info("ws connect %s  (now %d clients)", peer, len(_clients))

    # Before signaling, so the UI can render the toggle row even if the
    # operator has not asked for video yet.
    await websocket.send_text(json.dumps({
        "type": "camera_list",
        "cameras": [{"id": s.id, "label": s.label} for s in CAMERA_SPECS],
    }))

    types_seen: dict[str, int] = {}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type", "?")
            if t not in types_seen:
                logger.info("first %r msg from %s | keys=%s", t, peer, sorted(msg.keys()))
            types_seen[t] = types_seen.get(t, 0) + 1

            if t in RELAY_TYPES:
                await _broadcast(raw, exclude=websocket)
            elif t == "webrtc_request":
                await _handle_webrtc_request(state, msg)
            elif t == "webrtc_answer":
                await _handle_webrtc_answer(state, msg)
            elif t == "ice_candidate":
                # Dropped on purpose: aiortc gathers all local candidates before
                # offering, so on LAN the SDP candidates alone suffice.
                pass
            elif t == "camera_toggle":
                await _handle_camera_toggle(state, msg)
            elif t == "latency_report":
                # RTT stats from ?latency=1, logged so USB-vs-LAN comparisons
                # can be read off the workstation terminal. `host` self-labels.
                logger.info(
                    "latency_report host=%s n=%s mean=%.1f p50=%.1f p95=%.1f min=%.1f max=%.1f ms (one-way~%.1f)",
                    msg.get("host"), msg.get("n"),
                    msg.get("mean", 0.0), msg.get("p50", 0.0), msg.get("p95", 0.0),
                    msg.get("min", 0.0), msg.get("max", 0.0), msg.get("p50", 0.0) / 2.0,
                )
            else:
                await websocket.send_json({"echo": msg, "server_time": time.time()})

    except WebSocketDisconnect as e:
        logger.info("ws disconnect %s code=%s totals=%s", peer, e.code, types_seen)
    finally:
        await _close_pc(state)
        _clients.pop(websocket, None)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Use 0.0.0.0 for LAN access.")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--ssl-keyfile",  default=None, help="Path to TLS private key (PEM).")
    ap.add_argument("--ssl-certfile", default=None, help="Path to TLS cert chain (PEM).")
    ap.add_argument("--cameras-yaml", type=Path, help="Camera rig YAML; omit for pose-only relay.")
    args = ap.parse_args()

    if args.cameras_yaml is not None:
        if not args.cameras_yaml.is_file():
            ap.error(f"camera config does not exist: {args.cameras_yaml}")
        CAMERA_SPECS.extend(_load_camera_specs_from_yaml(args.cameras_yaml))
        logger.info("loaded %d camera(s) from %s", len(CAMERA_SPECS), args.cameras_yaml)

    import uvicorn
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info",
        ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
