"""Interactive browser viewport: MJPEG out, mouse in, over one TCP port.

Publishing frames is one-way; the return path lets the browser drive the camera:

    vp = Viewport(port=8080); vp.start()
    while True:
        cam = vp.take_camera()          # None until the user moves the mouse
        if cam:
            scene.set_orbit_camera(stage, **cam)
            vp.publish(renderer.frame(settle=1))

One TCP port, so VS Code's Remote-SSH forwards it automatically — see
DESIGN.md for why not Isaac's WebRTC livestream. Orbiting round-trips to the
workstation, throttled to one in-flight request, so a slow link degrades to a
lower update rate rather than a backlog of stale poses.

THE SERVER HOLDS NO CAMERA STATE. The client owns the orbit parameters and
posts the complete set every time, so a reconnecting tab restores its own view
and there is no drift from applying deltas.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY = "sparklabframe"

_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>sparklab_sim viewport</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0d0f12;color:#c9d1d9;
     font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
#wrap{position:fixed;inset:0;display:flex;flex-direction:column}
#bar{display:flex;gap:14px;align-items:center;padding:7px 12px;
     background:#161b22;border-bottom:1px solid #30363d;flex:0 0 auto}
#bar label{color:#8b949e}
select,button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
     border-radius:5px;padding:4px 9px;font:inherit;cursor:pointer}
button:hover,select:hover{background:#30363d}
#view{flex:1;position:relative;overflow:hidden;background:#000}
#img{position:absolute;inset:0;margin:auto;max-width:100%;max-height:100%;
     display:block;cursor:grab;user-select:none;-webkit-user-drag:none}
#img.drag{cursor:grabbing}
#hud{position:absolute;left:10px;bottom:10px;background:#0d1117d9;
     border:1px solid #30363d;border-radius:6px;padding:8px 11px;
     white-space:pre;pointer-events:none;font-size:12px}
#status{margin-left:auto;color:#8b949e}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
     background:#3fb950;margin-right:5px}
.dot.stale{background:#d29922}
#help{position:absolute;right:10px;bottom:10px;color:#6e7681;font-size:11px;
      text-align:right;pointer-events:none;white-space:pre}
</style></head><body><div id="wrap">
<div id="bar">
  <label>camera</label><select id="cam"></select>
  <button id="reset">reset view</button>
  <button id="copy">copy pose</button>
  <button id="reload">reload usd</button>
  <span id="status"><span class="dot"></span><span id="msg">connecting</span></span>
</div>
<div id="view">
  <img id="img" src="/stream" draggable="false">
  <div id="hud"></div>
  <div id="help">drag  orbit
shift-drag / middle  pan
wheel  zoom</div>
</div></div>
<script>
// Filled from /info on first poll — the server owns the opening view, derived
// from rig.py, so this file never carries a second copy of the geometry.
let HOME = null;
let cam = null;
let free = true;               // false => a camera authored in the USD
let inflight = false, queued = false;

const $ = id => document.getElementById(id);
const img = $('img'), hud = $('hud'), msg = $('msg'), dot = document.querySelector('.dot');

function hudText(){
  if(!free) return `camera  ${$('cam').value}\n(authored in the USD — not orbitable)`;
  if(!cam) return 'connecting…';
  const [x,y,z] = cam.target;
  return `azimuth   ${cam.azimuth_deg.toFixed(1)}°\n`
       + `elevation ${cam.elevation_deg.toFixed(1)}°\n`
       + `distance  ${cam.distance.toFixed(3)} m\n`
       + `target    ${x.toFixed(3)} ${y.toFixed(3)} ${z.toFixed(3)}\n`
       + `focal     ${cam.focal_mm.toFixed(1)} mm`;
}

// One request in flight at a time. A slow link lowers the update rate instead
// of queueing poses the user has already moved past.
function push(){
  if(!free || !cam) return;
  if(inflight){ queued = true; return; }
  inflight = true;
  fetch('/camera', {method:'POST', body:JSON.stringify(cam)})
    .then(() => { dot.classList.remove('stale'); msg.textContent = 'live'; })
    .catch(() => { dot.classList.add('stale'); msg.textContent = 'disconnected'; })
    .finally(() => {
      inflight = false;
      if(queued){ queued = false; push(); }
    });
  hud.textContent = hudText();
}

// --- orbit basis, mirrored from scene.set_orbit_camera --------------------
function basis(){
  const az = cam.azimuth_deg*Math.PI/180, el = cam.elevation_deg*Math.PI/180;
  // forward points from eye toward target
  const f = [-Math.cos(el)*Math.cos(az), -Math.cos(el)*Math.sin(az), -Math.sin(el)];
  let r = [f[1], -f[0], 0];                     // cross(forward, +Z)
  const rn = Math.hypot(r[0], r[1]) || 1;
  r = [r[0]/rn, r[1]/rn, 0];
  const u = [r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
  return [r, u];
}

let dragging = false, panning = false, lx = 0, ly = 0;
img.addEventListener('mousedown', e => {
  dragging = true; panning = e.shiftKey || e.button === 1;
  lx = e.clientX; ly = e.clientY; img.classList.add('drag'); e.preventDefault();
});
addEventListener('mouseup', () => { dragging = false; img.classList.remove('drag'); });
addEventListener('mousemove', e => {
  if(!dragging || !free || !cam) return;
  const dx = e.clientX - lx, dy = e.clientY - ly;
  lx = e.clientX; ly = e.clientY;
  if(panning){
    // Pan proportional to distance so the grab point tracks the cursor at any zoom.
    const s = cam.distance * 0.0022, [r,u] = basis();
    for(let i=0;i<3;i++) cam.target[i] += -dx*s*r[i] + dy*s*u[i];
  } else {
    cam.azimuth_deg   = (cam.azimuth_deg - dx*0.4) % 360;
    cam.elevation_deg = Math.max(-89, Math.min(89, cam.elevation_deg + dy*0.3));
  }
  push();
});
img.addEventListener('wheel', e => {
  if(!free || !cam) return;
  e.preventDefault();
  cam.distance = Math.max(0.05, Math.min(50, cam.distance * Math.exp(e.deltaY*0.0012)));
  push();
}, {passive:false});
img.addEventListener('contextmenu', e => e.preventDefault());

$('reset').onclick = () => { if(HOME){ cam = structuredClone(HOME); push(); } };
$('reload').onclick = () => fetch('/reload', {method:'POST'});
$('copy').onclick = () => {
  const t = free
    ? `scene.set_orbit_camera(stage, target=(${cam.target.map(v=>v.toFixed(4)).join(', ')}), `
      + `distance=${cam.distance.toFixed(4)}, azimuth_deg=${cam.azimuth_deg.toFixed(2)}, `
      + `elevation_deg=${cam.elevation_deg.toFixed(2)}, focal_mm=${cam.focal_mm.toFixed(1)})`
    : $('cam').value;
  navigator.clipboard.writeText(t).then(() => {
    msg.textContent = 'copied'; setTimeout(() => msg.textContent = 'live', 1200);
  });
};

$('cam').onchange = () => {
  const v = $('cam').value;
  free = (v === '__free__');
  fetch('/select', {method:'POST', body:JSON.stringify({camera:v})});
  hud.textContent = hudText();
};

async function info(){
  try{
    const r = await fetch('/info'); const d = await r.json();
    if(!HOME && d.home){            // first contact: adopt the server's view
      HOME = d.home; cam = structuredClone(HOME); hud.textContent = hudText(); push();
    }
    const sel = $('cam'), keep = sel.value;
    const want = ['__free__', ...d.cameras];
    if(sel.options.length !== want.length ||
       [...sel.options].some((o,i) => o.value !== want[i])){
      sel.innerHTML = '';
      for(const v of want){
        const o = document.createElement('option');
        o.value = v; o.textContent = (v === '__free__') ? 'free look' : v;
        sel.appendChild(o);
      }
      sel.value = want.includes(keep) ? keep : '__free__';
    }
    if(d.message){ msg.textContent = d.message; }
    dot.classList.remove('stale');
  }catch(e){ dot.classList.add('stale'); msg.textContent = 'disconnected'; }
}
info(); setInterval(info, 1500);
hud.textContent = hudText();
</script></body></html>"""


class Viewport:
    """MJPEG out plus a camera-pose return channel, on one port.

    The HTTP threads only touch lock-protected slots and the Isaac thread
    drains them. Nothing here calls into USD, which is render-thread-only.
    """

    def __init__(self, port: int = 8080, host: str = "127.0.0.1",
                 quality: int = 80, home: dict | None = None):
        self.port, self.host, self.quality = port, host, quality
        # The opening view, handed to the browser on its first /info poll so
        # the client carries no geometry constants of its own.
        self._home = dict(home) if home else None
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._camera: dict | None = None      # pending orbit pose
        self._selected: str | None = None     # pending camera switch
        self._reload = False                  # pending forced reload
        self._cameras: list[str] = []
        self._message = ""
        self._new_frame = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> str:
        if self._server is not None:
            return self.url
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                if not n:
                    return {}
                try:
                    return json.loads(self.rfile.read(n) or b"{}")
                except (ValueError, UnicodeDecodeError):
                    return {}

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE.encode())
                elif self.path == "/stream":
                    self._stream()
                elif self.path == "/info":
                    with outer._lock:
                        self._json({"cameras": list(outer._cameras),
                                    "message": outer._message,
                                    "home": outer._home})
                elif self.path == "/frame.jpg":
                    jpeg = outer._latest()
                    if jpeg is None:
                        self.send_error(503, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/camera":
                    d = self._body()
                    with outer._lock:
                        # Overwrite rather than queue: only the newest pose
                        # matters, and the user has already moved past the rest.
                        outer._camera = d
                    self._json({"ok": True})
                elif self.path == "/select":
                    with outer._lock:
                        outer._selected = self._body().get("camera")
                    self._json({"ok": True})
                elif self.path == "/reload":
                    with outer._lock:
                        outer._reload = True
                    self._json({"ok": True})
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
                self.end_headers()
                last = -1
                try:
                    while True:
                        jpeg, seq = outer._latest_with_seq()
                        if jpeg is None or seq == last:
                            outer._new_frame.wait(timeout=1.0)
                            continue
                        last = seq
                        self.wfile.write(
                            f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="viewport", daemon=True)
        self._thread.start()
        logger.info("viewport on %s", self.url)
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ---- frames out -------------------------------------------------------
    def publish(self, image: np.ndarray) -> None:
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {image.shape}")
        ok, buf = cv2.imencode(".jpg", image[..., ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        with self._lock:
            self._jpeg = buf.tobytes()
            self._seq += 1
        self._new_frame.set()
        self._new_frame.clear()

    def _latest(self):
        with self._lock:
            return self._jpeg

    def _latest_with_seq(self):
        with self._lock:
            return self._jpeg, self._seq

    # ---- input in ---------------------------------------------------------
    def take_camera(self) -> dict | None:
        """Newest orbit pose since the last call, or None if the user is idle.

        Drains rather than peeks, so no mouse movement means no work.
        """
        with self._lock:
            c, self._camera = self._camera, None
        if not c:
            return None
        # Only the keys set_orbit_camera takes, coerced — this is data off the
        # network and it is about to be handed to USD.
        try:
            return {
                "target": tuple(float(v) for v in c.get("target", (0, 0, 0.8))[:3]),
                "distance": float(c.get("distance", 2.2)),
                "azimuth_deg": float(c.get("azimuth_deg", -135.0)),
                "elevation_deg": float(c.get("elevation_deg", 25.0)),
                "focal_mm": float(c.get("focal_mm", 18.0)),
            }
        except (TypeError, ValueError, IndexError):
            return None

    def take_selection(self) -> str | None:
        with self._lock:
            s, self._selected = self._selected, None
        return s

    def take_reload(self) -> bool:
        with self._lock:
            r, self._reload = self._reload, False
        return r

    def set_info(self, cameras: list[str], message: str = "") -> None:
        """Publish the camera list and a status line to the browser."""
        with self._lock:
            self._cameras = list(cameras)
            self._message = message
