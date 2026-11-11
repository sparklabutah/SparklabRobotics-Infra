# rollout

Runs a policy like `lerobot-rollout`, but lets you change its task, pause it or
send the arms home while it runs, without reloading the model. Works on the real
arms and the Isaac twin.

```
live.py   the rollout loop and its /rollout/* interface
```

## Using it

```bash
./scripts/rollout.sh --mode=live

ros2 topic pub --once /rollout/task std_msgs/msg/String "{data: 'pick up the scissors'}"
ros2 service call /rollout/reset std_srvs/srv/Trigger
ros2 topic echo /rollout/status
```

Or use the web page: [`../harness/`](../harness/README.md).

| name | | |
|---|---|---|
| `/rollout/task` | topic in | new task; takes effect at once, dropping queued actions |
| `/rollout/reset` | service | send the arms home, clear queued actions, wait for a new task |
| `/rollout/home` | service | send the arms home and wait |
| `/rollout/park` | service | park the arms and wait |
| `/rollout/pause`, `/rollout/resume` | service | stop / continue stepping the policy |
| `/rollout/quit` | service | end the rollout |
| `/rollout/status` | topic out | JSON: task, paused, held, ticks, resets, cameras |
| `/rollout/state`, `/rollout/images/<cam>/compressed` | topic out | what the policy was shown |

Services answer at once; the loop carries them out between ticks. After `reset`,
`home` or `park` the arms hold still until you send a task or `resume`. Going home
is a smooth 5 s move (`--home-duration`, `--home`).
