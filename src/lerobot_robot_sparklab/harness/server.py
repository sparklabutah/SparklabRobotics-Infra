"""Web harness: put a high-level agent in front of the low-level VLA.

    sparklab-harness --rollout http://127.0.0.1:8090 --agent scripted
    sparklab-harness --agent gemini            # needs HARNESS_MODEL + GOOGLE_API_KEY

then open http://127.0.0.1:8099/ .

WHERE THIS SITS. It does not own the robot and never will::

    rollout_live process  (owns cameras, arms, the policy)
      :8090  GET /status  GET /observation  GET /frame/<cam>.jpg  POST /cmd
        ^
        | HTTP
    harness server (this file)     :8099
        |  serves the UI, runs the agent loop, proxies control
        v
    browser  +  high-level agent (in-process, or any external caller)

Two processes on purpose. The rollout holds a ~21 GB model and a live control
loop; restarting it costs ~90 s. Editing a prompt, swapping agents or reloading
the UI must not imply restarting that. It also means a crash in this file
cannot take the arms down — the worst case is the robot continuing on its last
instruction, which is why the loop only ever sets a task and the human keeps
pause and reset.

DRIVING IT FROM AN EXTERNAL AGENT. The built-in loop is a convenience, not the
interface. Anything that can speak HTTP can be the planner:

    curl -s :8099/api/state                      # frames available, task, ticks
    curl -s :8099/api/frame/top     -o top.jpg   # what the robot sees
    curl -s :8099/api/task -d '{"task":"pick up the scissors"}'

Run with --agent none and the harness is exactly that: a UI plus a REST facade
over the rollout, with nothing deciding on its own.
"""

# NO `from __future__ import annotations` in this file, deliberately. FastAPI
# resolves route annotations with get_type_hints(), which looks them up in the
# MODULE namespace -- but `Request` is imported inside build_app(). Under PEP 563
# the annotation is the string "Request", resolution fails, and FastAPI silently
# demotes the parameter to a query arg: every POST then 422s with
# {"loc":["query","req"],"msg":"Field required"} instead of reading the body.
# Without the future import the annotation is evaluated at def time, where
# `Request` is in scope.

import argparse
import json
import logging
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WEB = Path(__file__).parent / "web"


class Harness:
    """Agent loop + a cache of the rollout's state. All HTTP handlers read this."""

    def __init__(self, rollout: str, agent, goal: str = "", interval_s: float = 4.0):
        self.rollout = rollout.rstrip("/")
        self.agent = agent
        self.goal = goal
        self.interval_s = interval_s
        self.running = False
        self.history: list = []
        self.events: list[queue.Queue] = []      # one per connected browser
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        # Bumped on every stop. An agent turn takes a second or two (a model
        # call), and a task decided BEFORE the operator hit stop must not be
        # applied AFTER it -- that is a reset being silently undone by a
        # decision already in flight. step_once() compares this across the call.
        self._epoch = 0

    def stop_agent(self) -> None:
        """Stop the loop and invalidate any turn currently in flight."""
        with self._lock:
            self.running = False
            self._epoch += 1

    # ---- rollout transport -------------------------------------------------
    def _get(self, path: str, binary: bool = False):
        import urllib.request
        with urllib.request.urlopen(self.rollout + path, timeout=5) as r:
            return r.read() if binary else json.loads(r.read())

    def post_cmd(self, cmd: str, arg: str | None = None) -> dict:
        import urllib.request
        body = json.dumps({"cmd": cmd, "arg": arg}).encode()
        req = urllib.request.Request(self.rollout + "/cmd", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def status(self) -> dict:
        try:
            return self._get("/status")
        except Exception as e:
            return {"error": f"cannot reach rollout at {self.rollout}: {e}"}

    def observation(self) -> dict:
        try:
            return self._get("/observation")
        except Exception as e:
            return {"error": str(e), "cameras": [], "state": {}}

    def frame(self, cam: str) -> bytes | None:
        try:
            return self._get(f"/frame/{cam}.jpg", binary=True)
        except Exception:
            return None

    # ---- events ------------------------------------------------------------
    def emit(self, kind: str, **payload) -> None:
        ev = {"kind": kind, "at": time.time(), **payload}
        with self._lock:
            dead = []
            for q in self.events:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    dead.append(q)          # a browser tab that stopped reading
            for q in dead:
                self.events.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self.events.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self.events:
                self.events.remove(q)

    # ---- agent loop --------------------------------------------------------
    def step_once(self) -> dict:
        """One agent turn. Also the /api/agent {"action":"step"} handler."""
        epoch = self._epoch
        obs = self.observation()
        cams = obs.get("cameras") or []
        frames = {c: b for c in cams if (b := self.frame(c))}
        if not frames:
            d = {"error": "no frames from the rollout — is it running?"}
            self.emit("error", **d)
            return d

        decision = self.agent.decide(self.goal, frames, obs.get("state", {}), self.history)
        self.history.append(decision)
        self.last_error = decision.error

        applied = None
        if decision.task and self._epoch != epoch:
            # Stopped (or reset) while this turn was thinking. Dropping the task
            # is the whole point: applying it would drive the arms off the home
            # pose the operator just asked for.
            applied = {"skipped": "agent was stopped during this turn"}
        elif decision.task:
            # The one thing the agent is allowed to do. retarget() clears the
            # policy's queued chunk, so the change takes effect on the next
            # tick rather than after up to 30 stale actions.
            applied = self.post_cmd("task", decision.task)
        if decision.done:
            self.stop_agent()
            self.emit("done", reasoning=decision.reasoning)

        self.emit("decision", **decision.as_dict(), applied=applied)
        return {"ok": True, **decision.as_dict(), "applied": applied}

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self.running:
                continue
            try:
                self.step_once()
            except Exception as e:
                logger.exception("agent loop")
                self.last_error = f"{type(e).__name__}: {e}"
                self.emit("error", error=self.last_error)
                # Keep looping. Stopping here would leave the robot executing
                # its last instruction with nothing watching, which is worse
                # than a noisy retry the operator can see and act on.

    def start_thread(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="harness-agent")
        self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()


def build_app(h: Harness):
    from fastapi import FastAPI, Request
    from fastapi.responses import (FileResponse, JSONResponse, Response,
                                   StreamingResponse)

    app = FastAPI(title="SparkLab harness")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/state")
    def state():
        return {
            "rollout": h.status(),
            "observation": h.observation(),
            "agent": {
                "kind": getattr(h.agent, "name", "none") if h.agent else "none",
                "running": h.running,
                "goal": h.goal,
                "interval_s": h.interval_s,
                "last_error": h.last_error,
                "history": [d.as_dict() for d in h.history[-30:]],
            },
        }

    @app.get("/api/frame/{cam}")
    def frame(cam: str):
        blob = h.frame(cam)
        if blob is None:
            return JSONResponse({"error": f"no frame {cam!r}"}, status_code=404)
        return Response(blob, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    def _unreachable(e: Exception) -> JSONResponse:
        """A down rollout is the normal case while it does its ~90 s model load.

        Without this the urllib error escapes as a bare 500 "Internal Server
        Error" with no JSON body, and the UI can only report "HTTP 500" — which
        reads like a harness bug rather than "the thing you are driving is not
        up yet".
        """
        return JSONResponse(
            {"error": f"cannot reach rollout at {h.rollout}: {e}"}, status_code=502)

    @app.post("/api/task")
    async def set_task(req: Request):
        body = await req.json()
        task = (body.get("task") or "").strip()
        if not task:
            return JSONResponse({"error": "task must be non-empty"}, status_code=400)
        try:
            out = h.post_cmd("task", task)
        except Exception as e:
            return _unreachable(e)
        h.emit("manual", task=task)
        return out

    @app.post("/api/goal")
    async def set_goal(req: Request):
        body = await req.json()
        h.goal = (body.get("goal") or "").strip()
        h.history.clear()
        # Reset the agent too, not just the transcript. Clearing history alone
        # left a stateful agent mid-plan: it reported the OLD goal finished, so
        # Start armed the loop and the first turn immediately disarmed it.
        # Optional on the agent -- a stateless one needs nothing.
        reset = getattr(h.agent, "reset", None)
        if callable(reset):
            reset()
        h.stop_agent()
        h.last_error = None
        h.emit("goal", goal=h.goal)
        return {"ok": True, "goal": h.goal}

    @app.post("/api/cmd")
    async def cmd(req: Request):
        body = await req.json()
        c = (body.get("cmd") or "").strip().lower()
        # pause/resume/reset/park/home stay operator-only; the agent loop never
        # reaches this route. "quit" is excluded on purpose -- killing the
        # rollout from a web button is a footgun, and it owns the arms.
        if c not in {"pause", "resume", "reset", "park", "home"}:
            return JSONResponse({"error": f"{c!r} not allowed here"}, status_code=400)
        if c in {"pause", "reset"}:
            # Stop the agent too. A reset that leaves the loop running is undone
            # within one interval, when the next decision retargets the policy —
            # the arms ramp home and then immediately drive off again.
            h.stop_agent()
        try:
            return h.post_cmd(c, body.get("arg"))
        except Exception as e:
            return _unreachable(e)

    @app.post("/api/agent")
    async def agent_ctl(req: Request):
        body = await req.json()
        action = (body.get("action") or "").lower()
        if action == "start":
            if not h.agent:
                return JSONResponse({"error": "server started with --agent none"},
                                    status_code=400)
            if not h.goal:
                return JSONResponse({"error": "set a goal first"}, status_code=400)
            h.running = True
            h.emit("agent", running=True)
        elif action == "stop":
            h.stop_agent()
            h.emit("agent", running=False)
        elif action == "step":
            if not h.agent:
                return JSONResponse({"error": "no agent"}, status_code=400)
            return h.step_once()
        else:
            return JSONResponse({"error": "action: start|stop|step"}, status_code=400)
        return {"ok": True, "running": h.running}

    @app.get("/api/events")
    def events():
        def gen():
            q = h.subscribe()
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                h.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollout", default="http://127.0.0.1:8090",
                    help="control port of a running rollout_live")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--agent", default="scripted", choices=["scripted", "gemini", "none"])
    ap.add_argument("--goal", default="")
    ap.add_argument("--interval", type=float, default=4.0,
                    help="seconds between agent turns")
    ap.add_argument("--model", default=None, help="gemini model id (or HARNESS_MODEL)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    from . import agents
    agent = None
    if args.agent != "none":
        kw = {"model": args.model} if args.agent == "gemini" and args.model else {}
        agent = agents.build(args.agent, **kw)

    h = Harness(args.rollout, agent, goal=args.goal, interval_s=args.interval)
    h.start_thread()

    st = h.status()
    if "error" in st:
        logger.warning("rollout not reachable yet at %s — the UI will show the "
                       "error and recover when it appears", args.rollout)

    import uvicorn
    logger.info("harness UI on http://%s:%d  (rollout %s, agent %s)",
                args.host, args.port, args.rollout, args.agent)
    uvicorn.run(build_app(h), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
