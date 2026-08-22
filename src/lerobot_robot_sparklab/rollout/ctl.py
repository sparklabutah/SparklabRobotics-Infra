"""REPL client for a running ``lerobot_robot_sparklab.rollout.live`` loop.

    python -m lerobot_robot_sparklab.rollout.ctl                 # localhost:8090
    python -m lerobot_robot_sparklab.rollout.ctl --port 8090

Run it in a second pane, so typing is not fighting the rollout's scrolling log.

Commands:

    <anything else>      set the task to that text (the common case, no verb)
    task <text>          explicit form, for text that collides with a command
    reset                ramp arms home, clear the policy's action queue
    park                 ramp arms to the folded pose (server-side)
    home                 ramp arms home, leave the queue alone
    pause / resume       stop / restart stepping the policy
    status               current task, tick count, uptime
    quit                 stop the rollout process
    exit                 leave this REPL, rollout keeps running

One-shot from a script:

    python -m lerobot_robot_sparklab.rollout.ctl reset
    python -m lerobot_robot_sparklab.rollout.ctl "pick up the scissors"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

VERBS = {"task", "reset", "park", "pause", "resume", "home", "status", "quit"}


def call(base: str, cmd: str, arg: str | None = None, timeout: float = 15.0) -> dict:
    """Send one command. The reset/home ramp can take seconds, hence the timeout."""
    if cmd == "status":
        req = urllib.request.Request(base + "/status")
    else:
        body = json.dumps({"cmd": cmd, "arg": arg}).encode()
        req = urllib.request.Request(base + "/cmd", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"cannot reach {base}: {e.reason}. "
                                      f"Is rollout_live running with --control-port?"}


def parse(line: str) -> tuple[str, str | None] | None:
    """Map a REPL line to (cmd, arg).

    Bare text is a task, since retargeting is the common case. A line starting
    with a known verb is that verb — use the explicit 'task' form otherwise.
    """
    line = line.strip()
    if not line:
        return None
    head, _, rest = line.partition(" ")
    head = head.lower()
    if head in VERBS:
        return head, (rest.strip() or None)
    return "task", line


def show(resp: dict) -> None:
    if "task" in resp and "ticks" in resp:      # a /status payload
        print(f"  task    : {resp['task']!r}")
        print(f"  state   : {'PAUSED' if resp['paused'] else 'running'}")
        print(f"  ticks   : {resp['ticks']}   resets: {resp['resets']}"
              f"   uptime: {resp['uptime_s']}s")
        print(f"  last    : {resp['last_event']}")
    elif resp.get("ok"):
        print(f"  ok: {resp.get('queued', '')}")
    else:
        print(f"  ! {resp.get('error', resp)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("command", nargs="*",
                    help="run one command and exit; omit for a REPL")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    if args.command:
        parsed = parse(" ".join(args.command))
        if parsed is None:
            return 0
        show(call(base, *parsed))
        return 0

    try:
        import readline  # noqa: F401  -- history and line editing, if available
    except ImportError:
        pass

    print(f"connected to {base}   (bare text sets the task; 'exit' to leave)")
    show(call(base, "status"))
    while True:
        try:
            line = input("rollout> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().lower() in ("exit", "q"):
            return 0
        parsed = parse(line)
        if parsed is None:
            continue
        show(call(base, *parsed))
        if parsed[0] == "quit":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
