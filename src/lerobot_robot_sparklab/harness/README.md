# harness/

A high-level agent in front of the low-level VLA, plus a web UI to watch and
override it.

```
server.py   FastAPI: serves the UI, proxies the rollout, runs the agent loop
agents.py   the planners — ScriptedAgent (no API key) and GeminiAgent
web/        the UI, one self-contained HTML file
```

## The idea

MolmoAct2 takes a natural-language task and turns it into joint motion. It has
no memory and no planning: it executes the one instruction it currently holds.
`rollout_live.retarget()` already swaps that instruction cleanly — it clears the
policy's queued chunk, so a change lands on the next tick instead of after up to
30 stale actions.

So a high-level agent needs exactly one verb: **emit a new task string**.

```
goal "clear the table"
  -> agent looks at the camera frames
  -> "pick up the marker"      -> VLA -> arms
  -> "put it in the box"       -> VLA -> arms
  -> done
```

## Process split

The rollout owns the cameras, the arms and a ~21 GB model, and takes ~90 s to
start. The harness is a separate process so that editing a prompt, swapping
agents or reloading the UI does not imply restarting that — and so a crash here
cannot take the arms down.

```
rollout_live  :8090   GET /status  /observation  /frame/<cam>.jpg   POST /cmd
      ^ HTTP
harness       :8099   the UI, the agent loop, a REST facade
      ^
   browser
```

Frames are JPEG-encoded in the rollout's loop thread and kept latest-only, so a
slow watcher skips frames rather than building a backlog. Use
`--publish-every N` to subsample the encode off the control loop.

## Running it

```bash
./scripts/start_arm_servers.sh      # terminal 1 — owns CAN and torque
./scripts/rollout.sh --mode=live    # terminal 2 — policy + control port 8090
./scripts/harness.sh                # terminal 3 — UI on :8099
```

`--mode=live` is the ordinary rollout plus a control port. Everything else about
it is unchanged, so `--policy=stock`, `--task=...` and any `lerobot-rollout`
flag work exactly as they do without it.

`--agent scripted` (the default) walks a fixed subtask list with no API key. Use
it first: if the arm misbehaves under the scripted agent, the problem is not the
model.

`--agent gemini` needs `pip install google-genai`, `GOOGLE_API_KEY`, and
`HARNESS_MODEL=<model id>`. The model id is deliberately not defaulted — see
`GeminiAgent`'s docstring.

## Driving it from your own agent

The built-in loop is a convenience, not the interface. Run `--agent none` and
the harness is a UI plus a REST facade:

```bash
curl -s :8099/api/state                      # task, ticks, cameras, joint state
curl -s :8099/api/frame/top -o top.jpg       # what the robot sees
curl -s :8099/api/task -d '{"task":"pick up the scissors"}'
```

## The UI is three controls

Deliberately. On hardware `home`, `reset` and `park` all end in the same
`go_home` ramp — `park` has no sim transport to call, so it falls back to it —
so three separate buttons were three names for one behaviour.

| control | what it does |
|---|---|
| **Send task to VLA** | `POST /api/task` — retargets the policy now, bypassing the agent |
| **Send task to reasoning agent** | sets the goal and arms the agent loop, in one press |
| **Reset & park home** | stops the agent, ramps both arms home, and holds there until a task is sent |

Two things the reset button has to get right, both learned from the code it
drives:

*It stops everything that would drive the arm back out.* Two independent things
would: the agent loop, and the rollout itself. The agent is stopped first, and a
decision already in flight when you press it is dropped too — `step_once()`
compares an epoch counter across the agent call, so a task chosen *before* the
stop is never applied *after* it. The rollout then **holds** at home rather than
stepping the policy on the unchanged task; the status pill reads *parked — send
a task*, and sending one is what starts motion again.

*It waits for the real completion signal.* `POST /cmd` only queues; it returns
before the arms have moved, and the ramp takes up to `reset_timeout_s` (8 s) in
the rollout's loop thread. The button watches the rollout's own `resets`
counter, which `do_reset()` increments only once homing has finished, rather
than guessing with a timer.

## What an agent may do

Set a task, and declare the goal complete. That is all.

`/api/cmd` accepts only `pause`/`resume`/`reset`/`park`/`home` and the agent
loop never calls it; `quit` is rejected outright. This is not a security
boundary — anything that can retarget can already move the arms — it is a
blast-radius one. A confused agent picks a wrong subtask; it does not decide to
power down mid-motion.

The REST API still exposes `pause`, `resume`, `home`, `park` and the agent's
`start`/`stop`/`step` even though no button does. They are there for the REPL
(`sparklab-rollout-ctl`) and for driving the harness from your own code; the
UI just does not need a button per verb.
