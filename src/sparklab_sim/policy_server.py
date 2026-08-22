"""Serve the Isaac scene as a robot: joint commands in, observations out.

    ./scripts/isaac_python.sh -m sparklab_sim.policy_server

Boots Isaac once, loads ``scene.usda``, and exposes the bimanual rig over HTTP
in exactly the schema LeRobot records:

    POST /step   {"action": {"left_joint_1.pos": ..., ...}}  -> apply, then observe
    GET  /obs                                                 -> observe only
    GET  /health                                              -> readiness + shape

The matching LeRobot side is ``--robot.type=yam_ultra_sim`` (see
``robots/yam_ultra/sim_follower.py``), so a policy rollout against this needs
no policy changes at all -- same CLI, same observation keys, same units.

WHY A SERVER AND NOT A LIBRARY CALL
Isaac Sim runs under its own bundled Python 3.12, which has no ``lerobot``;
importing ``lerobot_robot_sparklab`` there fails immediately because its
package ``__init__`` imports lerobot to register the robot plugins. The two
cannot live in one interpreter, so they live in two and speak HTTP. This is
the same split the repo already uses for the arm servers.

/step IS ONE ROUND TRIP ON PURPOSE. A control tick is act-then-observe, and
splitting it into POST /action + GET /obs doubles the latency for no benefit.
GET /obs exists only for the first observation, before any action has been
sent.

ISAAC WORK RUNS ON THE MAIN THREAD, ALWAYS. ``app.update()`` is main-thread
affine: called from a request handler it does not raise, it simply never
returns, and the request hangs until the client's timeout while ``/health``
carries on answering perfectly — which makes it look like a network problem
rather than a threading one. So handlers do no Isaac work at all. They put a
job on a queue, block on an Event, and the main loop drains the queue, renders,
and wakes them. That is also why the main loop polls at 1 ms rather than
sleeping: it is the only thread that can service a tick.

UNITS, matching meta/stats.json of the recorded dataset:
  * joints  radians, in the URDF's own order and sign
  * gripper 0..1 normalised, 0 = closed -- which is already what
    ``scene.pose()`` takes, so nothing is rescaled here
  * images  uint8 HWC RGB, 640x360, wrist cameras center-cropped from
    640x480 exactly as ``cameras.yaml`` says the real D405 pipeline does
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _quat_wxyz_from_matrix(m):
    """Rotation of a row-vector 4x4 transform as (w, x, y, z).

    USD's ComputeLocalToWorldTransform is row-major with translation in the
    last ROW, so the rotation basis vectors are the first three rows -- and
    the matrix must be transposed before the usual column-convention
    extraction, or every prop comes back mirrored.
    """
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    r = [[r[j][i] for j in range(3)] for i in range(3)]      # -> column form
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x, y, z = (r[2][1] - r[1][2]) * s, (r[0][2] - r[2][0]) * s, (r[1][0] - r[0][1]) * s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2])
        w = (r[2][1] - r[1][2]) / s
        x, y, z = 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2])
        w = (r[0][2] - r[2][0]) / s
        x, y, z = (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1])
        w = (r[1][0] - r[0][1]) / s
        x, y, z = (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s
    return [w, x, y, z]

HANDS = ("left", "right")
ARM_JOINTS = 6

# name -> (camera prim short name, capture size, crop height or None).
# The wrist cameras capture 4:3 and are cropped, because that is what the
# hardware does: the D405 has no 640x360 mode at all, so the real rig grabs
# 640x480 and takes the middle 360 rows. Rendering 640x360 directly here
# would silently be a different vertical field of view.
CAMERAS = {
    "top": ("TopCamera", (640, 360), None),
    "left_wrist": ("LeftWristCamera", (640, 480), 360),
    "right_wrist": ("RightWristCamera", (640, 480), 360),
}


def _center_crop_height(frame, target_h: int):
    """Middle ``target_h`` rows. Mirrors follower._center_crop_height."""
    h = frame.shape[0]
    if h == target_h:
        return frame
    if h < target_h:
        raise ValueError(f"cannot crop {h} rows up to {target_h}")
    off = (h - target_h) // 2
    return frame[off:off + target_h]


class SimRig:
    """The Isaac-side rig. All USD/render calls happen on the main thread."""

    def __init__(self, app, scene_path: Path, settle: int, physics: bool = False):
        self.app = app
        self.scene_path = scene_path
        self.settle = settle
        self.physics = physics
        self._lock = threading.Lock()

        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import Articulation

        from . import render, scene

        self._scene = scene
        self._stage_utils = stage_utils

        # Same wrapper-stage trick as live.py: the editable file is a
        # sublayer, so nothing here writes to it.
        stage_utils.create_new_stage()
        st = stage_utils.get_current_stage()
        st.GetRootLayer().subLayerPaths = [str(scene_path)]
        app.update()
        scene.start(app)
        self.stage = stage_utils.get_current_stage()

        self.arms = {"left": Articulation(scene.LEFT_PRIM),
                     "right": Articulation(scene.RIGHT_PRIM)}

        if self.physics:
            # Drive targets do nothing without gains: the URDF ships none, so
            # every joint has stiffness 0 and a target exerts no force at all.
            for arm in self.arms.values():
                scene.set_drive_gains(arm)

        # Where every prop started, so /reset can put them back. Captured
        # BEFORE the first step -- once physics runs the props settle and
        # possibly get knocked about, and "the pose in the file" is the only
        # meaningful definition of a scene reset.
        self._prop_home = self._capture_props()

        found = scene.cameras(self.stage)
        self.renderers = {}
        for key, (prim_name, res, _crop) in CAMERAS.items():
            path = found.get(prim_name)
            if path is None:
                raise LookupError(
                    f"camera {prim_name!r} not in {scene_path}; found {list(found)}. "
                    f"The observation schema is fixed by the recorded dataset, so "
                    f"a missing camera is a hard error, not a warning.")
            self.renderers[key] = render.Renderer(app, path, resolution=res,
                                                  settle_frames=settle)
        # Seed a pose so the first observation is not whatever the file had.
        self.last_action = {h: [0.0] * ARM_JOINTS for h in HANDS}
        self.last_gripper = {h: 0.0 for h in HANDS}

    # ---- props -----------------------------------------------------------
    PROPS_ROOT = "/World/Props"

    def _prop_prims(self):
        """Every prop that is a dynamic rigid body, i.e. that can be moved."""
        from pxr import Usd, UsdPhysics

        root = self.stage.GetPrimAtPath(self.PROPS_ROOT)
        if not root.IsValid():
            return []
        return [p for p in Usd.PrimRange(root)
                if p.HasAPI(UsdPhysics.RigidBodyAPI)]

    def _capture_props(self) -> dict:
        from pxr import UsdGeom
        import numpy as np

        out = {}
        for p in self._prop_prims():
            x = UsdGeom.Xformable(p)
            m = x.ComputeLocalToWorldTransform(0)
            out[p.GetPath().pathString] = np.array(m, dtype=float)
        return out

    def reset_props(self) -> dict:
        """Put every prop back where the scene file had it, at rest.

        Zeroing velocity matters as much as the pose: a prop restored
        mid-flight keeps its momentum and immediately flies off again.
        """
        from isaacsim.core.experimental.prims import RigidPrim
        import numpy as np

        moved = []
        for path, m in self._prop_home.items():
            try:
                rp = RigidPrim(path)
                pos = m[3, :3]
                rot = _quat_wxyz_from_matrix(m)
                rp.set_world_poses(positions=[pos], orientations=[rot])
                rp.set_velocities(linear_velocities=[np.zeros(3)],
                                  angular_velocities=[np.zeros(3)])
                moved.append(path.rsplit("/", 1)[-1])
            except Exception as e:  # a prop may not be a tensor-API rigid body
                logging.warning("reset_props: %s -> %s: %s",
                                path, type(e).__name__, e)
        return {"reset": moved}

    # ---- parking ---------------------------------------------------------
    # All joints at zero is the YAM's folded rest pose (per the i2rt docs) and
    # it is where every recorded episode starts: across 49 demos the mean
    # start pose is left_joint_2 = 0.034, left_joint_3 = 0.022, and no episode
    # begins anywhere else. Leaving the arm elsewhere between runs puts the
    # NEXT rollout's first observation outside the training distribution --
    # measured, an unparked right arm sat 44 sigma off the demo start pose.
    PARK_Q = [0.0] * ARM_JOINTS
    PARK_GRIPPER = 0.0

    def current_q(self) -> dict:
        """Measured joint positions per hand, as (6 angles, gripper 0..1)."""
        import numpy as np

        out = {}
        for h in HANDS:
            try:
                q = np.asarray(self.arms[h].get_dof_positions()).reshape(-1)
                grip = (float(np.clip(abs(q[ARM_JOINTS]) / 0.04695, 0.0, 1.0))
                        if q.size > ARM_JOINTS else self.last_gripper[h])
                out[h] = (q[:ARM_JOINTS].astype(float).tolist(), grip)
            except Exception:
                out[h] = (list(self.last_action[h]), self.last_gripper[h])
        return out

    def is_parked(self, tol: float = 0.02) -> bool:
        return all(max(abs(v) for v in q) <= tol
                   for q, _g in self.current_q().values())

    def park(self, duration_s: float = 2.5, fps: float = 30.0) -> dict:
        """Ramp both arms to the folded pose, slowly, and stay there.

        Interpolated over ``duration_s`` rather than written in one go: a
        single write would teleport in kinematic mode and snap violently in
        physics mode, and this is meant to be watchable in the viewer -- it is
        the visible "run finished, workspace tidied" transition.

        Runs on the main thread like every other Isaac call, so it BLOCKS the
        service loop for its duration. That is intentional: nothing else
        should be driving the arm while it parks.
        """
        steps = max(int(duration_s * fps), 1)
        start = self.current_q()
        began = time.monotonic()
        for i in range(1, steps + 1):
            t = i / steps
            action = {}
            for h in HANDS:
                q0, g0 = start[h]
                for j in range(ARM_JOINTS):
                    action[f"{h}_joint_{j + 1}.pos"] = q0[j] * (1.0 - t) + self.PARK_Q[j] * t
                action[f"{h}_gripper.pos"] = g0 * (1.0 - t) + self.PARK_GRIPPER * t
            self.apply(action)
            self.app.update()
            # Pace to wall clock. Without this the ramp runs as fast as
            # app.update() happens to be -- measured 0.79 s for a requested
            # 2.0 s, because one update is ~13 ms, not the 33 ms the step
            # count assumes. `duration_s` should mean what it says, and the
            # whole point of the ramp is that it is watchable.
            behind = began + i / fps - time.monotonic()
            if behind > 0:
                time.sleep(behind)

        # Settle. The ramp leaves the joints with VELOCITY -- writing a
        # position does not clear it -- and the next few app.update() calls
        # integrate that forward, drifting the arm back off the parked pose.
        # Measured: up to 0.16 rad of residual once the ramp is paced to wall
        # clock (real elapsed time means a larger physics dt per update than
        # the unpaced version had). Zero the velocities, re-assert the pose,
        # and let it come to rest.
        import numpy as np

        park_action = {}
        for h in HANDS:
            for j in range(ARM_JOINTS):
                park_action[f"{h}_joint_{j + 1}.pos"] = self.PARK_Q[j]
            park_action[f"{h}_gripper.pos"] = self.PARK_GRIPPER
        for _ in range(5):
            for arm in self.arms.values():
                try:
                    n = np.asarray(arm.get_dof_velocities()).reshape(-1).size
                    arm.set_dof_velocities([np.zeros(n)])
                except Exception:
                    pass
            self.apply(park_action)
            self.app.update()

        end = self.current_q()
        residual = max(max(abs(v) for v in q) for q, _g in end.values())
        return {"parked": True, "steps": steps,
                "duration_s": duration_s,
                "max_residual_rad": round(residual, 5)}

    # ---- control ---------------------------------------------------------
    def apply(self, action: dict) -> None:
        """Send the commanded joint positions to both arms.

        Two modes, and they differ in whether the arm can touch anything:

        ``physics=False`` (default) writes joint state directly. The pose
        appears immediately and exactly, which is what a rendering/eval loop
        wants, but the fingers pass through objects -- a teleported body
        builds no contact force.

        ``physics=True`` writes PD drive targets instead. The arm is pulled
        toward the target over several steps, so it pushes things, and a
        closing gripper can hold them. The trade is that the arm LAGS the
        command and may not arrive at all against an obstruction -- which is
        the point when the sim stands in for hardware.
        """
        import numpy as np

        for h in HANDS:
            q = np.array([float(action[f"{h}_joint_{j}.pos"])
                          for j in range(1, ARM_JOINTS + 1)], dtype=float)
            g = float(action.get(f"{h}_gripper.pos", self.last_gripper[h]))
            send = self._scene.drive if self.physics else self._scene.pose
            send(self.arms[h], q, g)
            self.last_action[h] = q.tolist()
            self.last_gripper[h] = g

    def observe(self, settle: int | None = None) -> dict:
        import numpy as np

        state = {}
        for h in HANDS:
            # Read back from the articulation rather than echoing the command:
            # a joint driven past its limit clamps, and the policy should see
            # where the arm actually is.
            try:
                q = np.asarray(self.arms[h].get_dof_positions()).reshape(-1)
            except Exception:
                q = np.array(self.last_action[h] + [self.last_gripper[h]] * 2)
            for j in range(1, ARM_JOINTS + 1):
                state[f"{h}_joint_{j}.pos"] = float(q[j - 1])
            # Fingers are prismatic, -0.04695..0 in the URDF, 0 closed. The
            # dataset records gripper as 0..1, so map back the same way
            # scene.pose() maps forward.
            if q.size > ARM_JOINTS:
                state[f"{h}_gripper.pos"] = float(
                    np.clip(abs(q[ARM_JOINTS]) / 0.04695, 0.0, 1.0))
            else:
                state[f"{h}_gripper.pos"] = self.last_gripper[h]

        # ONE pump for ALL cameras. Every render product advances on the same
        # app.update(), so pumping per camera does the same global work three
        # times over: at settle=2 that is 6 updates a tick where 2 will do.
        # Measured, this is the difference between ~116 ms and ~40 ms per
        # observation. Hence frame(settle=0) below -- read, do not pump.
        n = self.settle if settle is None else settle
        for _ in range(n):
            self.app.update()

        images = {}
        for key, (_p, _res, crop) in CAMERAS.items():
            frame = self.renderers[key].frame(settle=0)
            if crop is not None:
                frame = _center_crop_height(frame, crop)
            images[key] = frame
        return {"state": state, "images": images}

    def close(self) -> None:
        for r in self.renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self.renderers.clear()


def _encode(images: dict, quality: int) -> dict:
    import cv2

    out = {}
    for k, arr in images.items():
        ok, buf = cv2.imencode(".jpg", arr[..., ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError(f"JPEG encode failed for {k}")
        out[k] = base64.b64encode(buf.tobytes()).decode("ascii")
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--settle", type=int, default=2,
                    help="app updates per rendered frame. 2 is enough: this "
                         "scene is converged at settle=1 (measured "
                         "frame-to-frame MAE 0.02-0.04), and every extra "
                         "update costs ~10 ms across three cameras")
    ap.add_argument("--physics", action="store_true",
                    help="drive the arms with PD position targets instead of "
                         "writing joint state directly. Needed for anything "
                         "that touches an object: a teleported finger builds "
                         "no contact force and passes straight through. Costs "
                         "exactness -- the arm now LAGS the command and can "
                         "be stopped by an obstruction, like real hardware. "
                         "OFF by default so existing eval runs, which depend "
                         "on the commanded pose being reached exactly, are "
                         "unchanged")
    ap.add_argument("--auto-park-s", type=float, default=45.0,
                    help="ramp the arms home after this many seconds with no "
                         "ACTION received (0 disables). Covers the case a "
                         "clean disconnect cannot: a rollout that is killed "
                         "or crashes never parks itself, and the pose it "
                         "abandons becomes the NEXT run's captured 'initial "
                         "position' -- measured, that left one arm 44 sigma "
                         "outside the demo start distribution. Polling /obs "
                         "does not count as activity; only /step does")
    ap.add_argument("--park-seconds", type=float, default=2.5,
                    help="how long the park ramp takes. Slow on purpose so it "
                         "is watchable and, under --physics, does not snap")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="seconds a queued step may take before the handler "
                         "gives up on the main loop")
    ap.add_argument("--quality", type=int, default=90,
                    help="JPEG quality for observation images. High on "
                         "purpose -- these feed a policy, and compression "
                         "artefacts are a distribution shift it never saw")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    from . import paths

    scene_path = Path(args.scene) if args.scene else (
        paths.YAM_ULTRA_MODEL_DIR.parent / "sim" / "scene.usda")
    if not scene_path.exists():
        print(f"no scene at {scene_path}")
        return 1

    print(f"  scene : {scene_path}")
    t0 = time.time()
    rig = SimRig(app, scene_path, args.settle, physics=args.physics)
    print(f"  rig   : ready in {time.time() - t0:.1f}s, "
          f"cameras {list(rig.renderers)}")

    stats = {"steps": 0, "render_s": 0.0}
    # Handlers put work here; only the main loop takes it out. See the module
    # docstring for why this indirection is not optional.
    jobs: "queue.Queue[dict]" = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, UnicodeDecodeError):
                return {}

        def do_GET(self):
            if self.path == "/health":
                self._json({"ok": True,
                            "cameras": {k: [CAMERAS[k][1][0],
                                            CAMERAS[k][2] or CAMERAS[k][1][1]]
                                        for k in CAMERAS},
                            "hands": list(HANDS),
                            "arm_joints": ARM_JOINTS,
                            "steps": stats["steps"]})
            elif self.path == "/obs":
                self._json(self._observe())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/park":
                # Ramps the arms home. Slow by design, so give it room: the
                # default request timeout is shorter than the ramp itself.
                try:
                    body = self._body()
                    self._json(self._submit(None, park=True,
                                            park_s=float(body.get("duration_s", 2.5))))
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, code=500)
                return
            if self.path == "/reset":
                # Same main-thread rule as /step: touching prop transforms is
                # Isaac work, so it goes through the queue, not this thread.
                try:
                    self._json(self._submit(None, reset=True))
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, code=500)
                return
            if self.path != "/step":
                self.send_error(404)
                return
            try:
                self._json(self._submit((self._body().get("action") or {})))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, code=500)

        def _observe(self):
            return self._submit(None)

        def _submit(self, action, reset: bool = False,
                    park: bool = False, park_s: float = 2.5):
            """Hand the work to the main thread and wait for it.

            Nothing Isaac-related may run here — see the module docstring.
            """
            job = {"action": action, "reset": reset, "park": park,
                   "park_s": park_s,
                   "done": threading.Event(), "result": None}
            jobs.put(job)
            # A park ramp deliberately takes seconds; waiting only `timeout`
            # would report a spurious failure while the ramp is still running.
            wait = args.timeout + (park_s + 5.0 if park else 0.0)
            if not job["done"].wait(timeout=wait):
                return {"error": f"sim step did not complete in {wait}s"}
            return job["result"]

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"  serve : http://{args.host}:{args.port}/")
    print("  ready — run the rollout with --robot.type=yam_ultra_sim\n")

    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stopping.set())

    # Idle auto-park. A client that exits CLEANLY parks itself on disconnect,
    # but one that is killed, crashes or loses the connection never gets to --
    # and that is exactly the case that poisons the next run, because the
    # abandoned pose becomes the next rollout's captured "initial position".
    # So the workspace also tidies itself: no ACTION for this long and the
    # arms ramp home on their own.
    #
    # Only /step counts as activity. Polling /obs is watching, not driving,
    # and must not keep a dead rollout's pose alive indefinitely.
    last_action_at = time.monotonic()
    auto_parked = False

    try:
        # The service loop. This thread is the only one allowed to touch
        # Isaac, so it must stay responsive: poll at 1 ms rather than sleeping
        # in 200 ms chunks, or every control tick inherits that latency.
        while not stopping.is_set():
            try:
                job = jobs.get(timeout=0.001)
            except queue.Empty:
                if (args.auto_park_s > 0 and not auto_parked
                        and time.monotonic() - last_action_at > args.auto_park_s
                        and not rig.is_parked()):
                    print(f"  idle {args.auto_park_s:.0f}s with no action — "
                          f"parking arms", flush=True)
                    try:
                        info = rig.park(duration_s=args.park_seconds)
                        print(f"  parked (residual "
                              f"{info['max_residual_rad']:.4f} rad)", flush=True)
                    except Exception as e:
                        print(f"  auto-park failed: {type(e).__name__}: {e}",
                              flush=True)
                    auto_parked = True
                continue
            try:
                t = time.time()
                if job.get("park"):
                    job["result"] = rig.park(duration_s=job.get("park_s", 2.5))
                    auto_parked = True
                    continue
                if job["action"]:
                    last_action_at = time.monotonic()
                    auto_parked = False
                if job.get("reset"):
                    rig.reset_props()
                if job["action"]:
                    rig.apply(job["action"])
                obs = rig.observe()
                job["result"] = {"state": obs["state"],
                                 "images": _encode(obs["images"], args.quality)}
                if job.get("reset"):
                    job["result"]["reset"] = sorted(
                        p.rsplit("/", 1)[-1] for p in rig._prop_home)
                stats["render_s"] += time.time() - t
                stats["steps"] += 1
            except KeyError as e:
                job["result"] = {"error": f"action missing key {e}"}
            except Exception as e:
                job["result"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                job["done"].set()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if stats["steps"]:
            print(f"  {stats['steps']} steps, "
                  f"{stats['render_s'] / stats['steps'] * 1000:.1f} ms/obs avg")
        server.shutdown()
        server.server_close()
        rig.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
