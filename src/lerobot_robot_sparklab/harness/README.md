# harness

A high-level agent in front of the low-level VLA, plus a web UI to watch and
override it.

```
server.py   FastAPI: serves the UI, proxies the rollout, runs the agent loop
agents.py   the planners — ScriptedAgent (no API key) and GeminiAgent
web/        the UI, one self-contained HTML file
```

The VLA takes a natural-language task and turns it into joint motion, holding
one instruction at a time. `rollout_live.retarget()` swaps that instruction
cleanly, so a high-level agent needs exactly one verb: emit a new task string.

```
goal "clear the table"
  -> agent looks at the camera frames
  -> "pick up the marker"      -> VLA -> arms
  -> "put it in the box"       -> VLA -> arms
  -> done
```

## Process split

```
rollout_live  :8090   GET /status  /observation  /frame/<cam>.jpg   POST /cmd
      ^ HTTP
harness       :8099   the UI, the agent loop, a REST facade
      ^
   browser
```

The rollout owns the cameras, the arms and the model; the harness owns nothing
and can be restarted freely. See
[`DESIGN.md`](../../../DESIGN.md#the-agent-harness).

Frames are JPEG-encoded in the rollout's loop thread and kept latest-only, so a
slow watcher skips frames rather than building a backlog. `--publish-every N`
subsamples the encode off the control loop.

## Running it

```bash
./scripts/start_arm_servers.sh      # terminal 1 — owns CAN and torque
./scripts/rollout.sh --mode=live    # terminal 2 — policy + control port 8090
./scripts/harness.sh                # terminal 3 — UI on :8099
```

`--mode=live` is the ordinary rollout plus a control port, so `--policy=stock`,
`--task=...` and any `lerobot-rollout` flag work as they do without it.

`--agent scripted` (the default) walks a fixed subtask list with no API key.
Use it first: if the arm misbehaves under the scripted agent, the problem is
not the model.

`--agent gemini` needs `pip install google-genai`, `GOOGLE_API_KEY`, and
`HARNESS_MODEL=<model id>`.

## Driving it from your own agent

Run `--agent none` and the harness is a UI plus a REST facade:

```bash
curl -s :8099/api/state                      # task, ticks, cameras, joint state
curl -s :8099/api/frame/top -o top.jpg       # what the robot sees
curl -s :8099/api/task -d '{"task":"pick up the scissors"}'
```

## The UI's three controls

| control | what it does |
|---|---|
| **Send task to VLA** | `POST /api/task` — retargets the policy now, bypassing the agent |
| **Send task to reasoning agent** | sets the goal and arms the agent loop, in one press |
| **Reset & park home** | stops the agent, ramps both arms home, holds until a task is sent |

Reset stops the agent loop *and* the rollout: a decision already in flight when
you press it is dropped, and the rollout then holds at home rather than
stepping the policy on the unchanged task. The status pill reads *parked — send
a task*. It watches the rollout's `resets` counter, which increments only once
homing has finished, rather than guessing with a timer.

On hardware `home`, `reset` and `park` all end in the same `go_home` ramp, which
is why there are three controls rather than six.

## What an agent may do

Set a task, and declare the goal complete. That is all. `/api/cmd` accepts only
`pause`/`resume`/`reset`/`park`/`home`, the agent loop never calls it, and
`quit` is rejected outright.

The REST API still exposes `pause`, `resume`, `home`, `park` and the agent's
`start`/`stop`/`step` even though no button does — they are there for
`sparklab-rollout-ctl` and for driving the harness from your own code.
