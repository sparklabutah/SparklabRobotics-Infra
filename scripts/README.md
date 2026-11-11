# scripts

Everyday commands. Run them from the repo root, with the `sparklab-ros` env active
and the workspace sourced.

| script | does |
|---|---|
| `rollout.sh` | run a policy, on the real arms or in Isaac |
| `record_manual.sh` | record a dataset; Space or B/Y starts and ends episodes |
| `record.sh` | record a dataset with fixed-length episodes (`lerobot-record`) |
| `harness.sh` | web page + AI agent in front of a running policy |
| `make_certs.sh` | the HTTPS certificate the headset needs |
| `can_health.sh` | check the CAN bus when an arm stops responding |
| `isaac_python.sh` | run something with Isaac Sim's Python ([sparklab_sim](../src/sparklab_sim/README.md)) |
| `capx_python.sh` | run something with cap-x's GPU stack and ROS 2 ([capx_sparklab](../src/capx_sparklab/README.md)) |
| `install_isaac_kernel.sh`, `start_isaac_jupyter.sh` | Isaac notebooks in VS Code |
| `analysis/*.py` | find out why a policy misbehaves |

Everything below needs the rig started first, in its own terminal:

```bash
ros2 launch sparklab_yam yam.launch.py                     # real arms
ros2 launch sparklab_yam yam.launch.py sim:=true viewer:=true cameras:=false   # no hardware
```

## Record a dataset

Start the rig *without* `teleop:=true`; the recorder brings its own. Then:

```bash
./scripts/record_manual.sh      # Space / B/Y start and end, r discards, q quits
./scripts/record.sh             # fixed 30 s episodes
```

Set the dataset name and task inside the script, or pass `--dataset.*` flags.

## Run a policy

```bash
./scripts/rollout.sh                          # fine-tuned checkpoint on the real arms
./scripts/rollout.sh --policy=stock-typed     # the stock checkpoint, for comparison
./scripts/rollout.sh --mode=live              # ...and change its task while it runs
./scripts/rollout.sh --mode=dagger            # take over with the Quest to correct it
./scripts/rollout.sh --dry-run                # print the command only
```

Other flags pass through to `lerobot-rollout` (e.g. `--duration=120`).

**In Isaac instead:**

```bash
./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics   # terminal 1
./scripts/rollout.sh --robot=sim --policy=stock                                 # terminal 2
```

**If the arm moves sluggishly**, look for `loop running slower than target` in the
log and try `--no-foxglove`.

## Find out why a policy misbehaves

Run these in order; each rules out one layer.

```bash
# 1. Is the data pipeline right? (compares the policy to the recorded human action)
python scripts/analysis/validate_policy_pipeline.py --checkpoint <ckpt>/pretrained_model
# 2. Does it follow the instruction? (--in-scene: only visible objects)
python scripts/analysis/lang_probe.py --checkpoint <ckpt>/pretrained_model --in-scene
# 3. Is it jerky?
python scripts/analysis/compare_action_noise.py --checkpoints <ckpt1> <ckpt2> --episodes 0 5 12 --frames 2
# 4. Does it lunge toward the rest pose?
python scripts/analysis/inspect_policy_chunks.py --checkpoint <ckpt>/pretrained_model --episode 5
# 5. Does the sim look different enough to matter? (needs the Isaac server)
python scripts/analysis/compare_real_vs_sim_images.py --checkpoint <ckpt>/pretrained_model --port 8081
```

`profile_rollout.py` times each stage of a slow loop; it takes the `lerobot-rollout`
flags plus `--ticks`.

## Make the headset certificate

```bash
./scripts/make_certs.sh                  # every IP of this machine
./scripts/make_certs.sh 10.195.64.139    # or only the ones you name
```

It writes `cert.pem` and `key.pem` in the repo root, where the launch looks. The
headset must accept a new certificate once, so don't regenerate it casually.

## Good to know

- A rollout (~13 GB), the Isaac server (~4 GB) and an analysis script (~13–27 GB)
  won't all fit in the GPU's 32 GB at once.
