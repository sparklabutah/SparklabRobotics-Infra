"""On-demand capture server: click a button, render the rig, view the frames.

    ./scripts/isaac_python.sh -m sparklab_sim.capture

Each click renders a high-quality workspace overview plus the three rig cameras
at their real resolutions and fields of view. Nothing renders between clicks,
so idle costs nothing and each capture can afford enough settle frames to be
clean — unlike a live stream, which must stay responsive.

``app.update()`` must run on the main thread, so the HTTP handler sets a
request flag and waits while the main loop renders and signals completion.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGE = """<!doctype html><html><head><title>sparklab_sim capture</title>
<style>
 body{margin:0;background:#0e0e10;color:#ddd;font:14px system-ui;padding:16px}
 h1{font-size:15px;font-weight:600;margin:0 0 12px}
 button{background:#2d6cdf;color:#fff;border:0;padding:10px 20px;border-radius:6px;
        font-size:14px;cursor:pointer}
 button:disabled{background:#444;cursor:wait}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;margin-top:16px}
 figure{margin:0;background:#181820;border-radius:8px;padding:8px}
 figcaption{font-size:12px;color:#9aa;padding:4px 2px}
 img{width:100%;height:auto;display:block;border-radius:4px}
 #status{margin-left:12px;color:#9aa;font-size:13px}
</style></head><body>
<h1>sparklab_sim &mdash; capture</h1>
<button id="go" onclick="cap()">Capture</button><span id="status">ready</span>
<div class="grid" id="grid"></div>
<script>
async function cap(){
  const b=document.getElementById('go'), s=document.getElementById('status');
  b.disabled=true; s.textContent='rendering...';
  const t0=performance.now();
  try{
    const r=await fetch('/capture',{method:'POST'});
    if(!r.ok) throw new Error(await r.text());
    const j=await r.json();
    const g=document.getElementById('grid'); g.innerHTML='';
    for(const v of j.views){
      const f=document.createElement('figure');
      f.innerHTML='<img src="/view/'+v.name+'?t='+Date.now()+'">'+
                  '<figcaption>'+v.name+' &mdash; '+v.w+'x'+v.h+
                  (v.note?' &mdash; '+v.note:'')+'</figcaption>';
      g.appendChild(f);
    }
    s.textContent='done in '+((performance.now()-t0)/1000).toFixed(1)+'s';
  }catch(e){ s.textContent='error: '+e.message; }
  b.disabled=false;
}
cap();
</script></body></html>"""


class CaptureServer:
    """HTTP front end. Renders nothing itself — see the module docstring."""

    def __init__(self, port: int = 8890, host: str = "127.0.0.1"):
        self.port, self.host = port, host
        self._lock = threading.Lock()
        self._views: dict[str, tuple[bytes, int, int, str]] = {}
        self._want = threading.Event()
        self._done = threading.Event()
        self._error: str | None = None
        self._server = None

    # ---- called from the main loop ----------------------------------------
    def requested(self) -> bool:
        return self._want.is_set()

    def publish(self, views: dict, error: str | None = None) -> None:
        """Store rendered views: ``{name: (rgb, note)}``. Signals the waiter."""
        import cv2

        with self._lock:
            self._error = error
            if error is None:
                self._views = {}
                for name, (img, note) in views.items():
                    ok, buf = cv2.imencode(".jpg", img[..., ::-1],
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if ok:
                        self._views[name] = (buf.tobytes(), img.shape[1],
                                             img.shape[0], note)
        self._want.clear()
        self._done.set()

    # ---- server -----------------------------------------------------------
    def start(self) -> str:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, ctype, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, "text/html", _PAGE.encode())
                elif self.path.startswith("/view/"):
                    name = self.path[len("/view/"):].split("?")[0]
                    with outer._lock:
                        v = outer._views.get(name)
                    if v is None:
                        self.send_error(404, f"no view {name!r}")
                    else:
                        self._send(200, "image/jpeg", v[0])
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/capture":
                    self.send_error(404)
                    return
                # Hand the work to the main thread and wait for it.
                outer._done.clear()
                outer._want.set()
                if not outer._done.wait(timeout=180):
                    self.send_error(504, "render timed out")
                    return
                with outer._lock:
                    if outer._error:
                        self._send(500, "text/plain", outer._error.encode())
                        return
                    import json
                    body = json.dumps({"views": [
                        {"name": n, "w": w, "h": h, "note": note}
                        for n, (_, w, h, note) in outer._views.items()]}).encode()
                self._send(200, "application/json", body)

        self._server = ThreadingHTTPServer((self.host, self.port), H)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever,
                         name="capture-http", daemon=True).start()
        return self.url

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--scene", default=None, help="scene .usda to open (optional)")
    ap.add_argument("--overview-res", default="1600x900")
    ap.add_argument("--settle", type=int, default=24,
                    help="app updates per frame; high is fine, this is on demand")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import numpy as np
    import isaacsim.core.experimental.utils.stage as stage_utils

    from . import render, rig, rigcams, scene

    ow, oh = (int(v) for v in args.overview_res.lower().split("x"))

    if args.scene:
        stage_utils.open_stage(args.scene)
        scene.start(app)
        left = right = None
    else:
        left, right = scene.build()
        scene.start(app)

    stage = stage_utils.get_current_stage()
    specs = rigcams.load_specs()
    cams = rigcams.add_rig_cameras(stage, specs)
    scene.add_overview_camera()

    if left is not None:
        scene.pose(left, np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0]), 0.0)
        scene.pose(right, np.array([0.3, 1.2, 1.0, 0.0, 0.4, 0.0]), 1.0)

    print(rig.summary())
    print(rigcams.summary(specs))

    # One renderer per view, built once. Overview is deliberately larger.
    renderers = {"overview": (render.Renderer(app, scene.OVERVIEW_PRIM, (ow, oh),
                                              settle_frames=args.settle),
                              "workspace overview (authoring view, not a rig camera)")}
    for cid, info in cams.items():
        note = "PLACEHOLDER mount" if cid != "top" else "standoff/height ASSUMED"
        renderers[cid] = (render.Renderer(app, info["path"], info["resolution"],
                                          settle_frames=args.settle), note)

    srv = CaptureServer(port=args.port)
    print(f"\n  capture UI: {srv.start()}\n  Ctrl-C to stop\n")

    try:
        while True:
            if srv.requested():
                t0 = time.time()
                try:
                    views = {n: (r.frame(), note) for n, (r, note) in renderers.items()}
                    srv.publish(views)
                    print(f"[{time.strftime('%H:%M:%S')}] captured "
                          f"{len(views)} views in {time.time() - t0:.1f}s")
                except Exception as e:
                    srv.publish({}, error=repr(e))
                    print(f"  capture failed: {e!r}")
            else:
                # Idle: keep Kit alive cheaply. Nothing renders.
                app.update()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.stop()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
