# scripts/

Every command below is copy-pasteable from the repo root.

**Shell scripts** are executable and anchor themselves to the repo root, so
they work from any directory:

    ./scripts/rollout.sh

**Python scripts** are not executable, deliberately — each needs a specific
interpreter, and a bare `./script.py` would pick up whatever `python3` is first
on `PATH`, which has no torch. There is one invocation for all of them:

    PY="PYTHONPATH=src /opt/miniforge/envs/robot-py312/bin/python"

`robot-py312` is the env with **lerobot 0.6.0**, which is what `molmoact2` and
`pi05` need. The `robot` and `teleop` envs are on 0.4.4 and cannot load these
checkpoints — that mismatch is the most common cause of a confusing failure here.

---

# I want to…

## …run the robot

**1. Start the arm servers. Always first** — they own the CAN loop and the
torque; everything else is a client of them.

    ./scripts/start_arm_servers.sh              # real hardware
    ./scripts/start_arm_servers.sh --sim        # i2rt SimRobots, nothing moves

Ports and CAN channels are overridable:
`LEFT_CHANNEL=can0 RIGHT_CHANNEL=can1 LEFT_PORT=11333 RIGHT_PORT=11334`.

**2. If an arm stops responding**, before anything else:

    ./scripts/can_health.sh

**3. Teleoperate**, no recording:

    ./scripts/teleop_demo.sh

**4. Record a dataset.**

    ./scripts/record.sh

## …run a policy

`rollout.sh` is the entry point; robot, policy and strategy are separate flags.

| I want | command |
|---|---|
| a fine-tuned checkpoint on the real arms | `./scripts/rollout.sh` |
| the stock base checkpoint, for comparison | `./scripts/rollout.sh --policy=stock-typed` |
| DAgger corrections (teleop takeover) | `./scripts/rollout.sh --mode=dagger` |
| a policy in sim | `./scripts/rollout.sh --robot=sim --policy=stock` |
| a policy in sim, retargetable while running | `./scripts/rollout.sh --robot=sim --mode=live` |
| a policy on hardware, retargetable while running | `./scripts/rollout.sh --mode=live` |
| a high-level agent driving the VLA | `--mode=live`, then `./scripts/harness.sh` |
| to see the command without running it | add `--dry-run` |

Unrecognised flags pass through to `lerobot-rollout`, so
`--duration=120 --policy.num_flow_timesteps=4` works with no edit. `rollout.sh`
checks that the arm servers / sim server / relay it needs are actually listening
and names whichever is missing, instead of timing out inside `connect()`.

It is one self-contained file — robot, policy, strategy and runner are four
blocks you can read top to bottom. Change rollout behaviour there.

Sim rollouts need the Isaac server up first, in its own terminal:

    ./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081

Add `--physics` to that for anything involving **contact**. Without it the
fingers teleport and cannot hold an object, so every grasp fails regardless of
the policy. It is left running between rollouts on purpose (Isaac costs ~15 s
to boot) and parks the arms on its own after 45 s idle.

Running two rollouts at once? Foxglove aborts on a bound port rather than
falling back, so move both:

    FOXGLOVE_PORT=8766 CONTROL_PORT=8091 ./scripts/rollout.sh --robot=sim --mode=live

**Driving a live rollout** (`--mode=live`, on hardware or sim) from a second
pane — retargeting clears the policy's queued chunk, so a new instruction lands
on the next tick rather than after up to 30 stale actions:

    sparklab-rollout-ctl

    rollout> pick up the scissors     # bare text retargets the policy
    rollout> reset                    # ramp home, clear the action queue
    rollout> park / pause / resume / status

Or drive it from a browser and a high-level agent — see
[`../src/lerobot_robot_sparklab/harness/README.md`](../src/lerobot_robot_sparklab/harness/README.md):

    ./scripts/rollout.sh --mode=live      # terminal 2
    ./scripts/harness.sh                  # terminal 3 -> http://127.0.0.1:8099

### Two things that will cost you an afternoon

**Do not re-specify a checkpoint's inference config.** A `--policy.path` load
already carries dtype, chunking, cuda graphs and amp in its `config.json`, and
`rollout.sh` deliberately adds nothing to it. Forcing `--policy.use_amp=true`
onto a checkpoint that says `use_amp: false` wraps a bfloat16 model in fp16
autocast (that flag *is* `torch.autocast(device_type="cuda")`, which defaults
to float16) — slower, and worse actions. Only `--policy=stock-typed` spells the
config out, because `--policy.type` infers nothing.

**A slow loop is a slow arm.** One tick consumes one action from a 30 fps
trajectory, so missing the fps target plays it in slow motion. If the arm feels
sluggish, look for `loop running slower than target` and try:

    CPU_CORES= ./scripts/rollout.sh --mode=live --no-foxglove

`--display_data` logs three camera images every tick, and the default
`taskset -c 0-5` reserves 6 of 24 cores.

## …work out why a policy is behaving badly

Run these in order — each rules out a layer, so a failure early on makes the
later ones meaningless.

**1. Is the pipeline itself correct?** Feeds recorded frames through the exact
processors `lerobot-rollout` builds and compares against the human action. A
large error here means an image-key or normalisation bug, not a bad policy.

    $PY scripts/analysis/validate_policy_pipeline.py --checkpoint <ckpt>/pretrained_model

**2. Does the policy read the instruction?** Swaps which object is named and
measures the response against paraphrase and sampling noise.

    $PY scripts/analysis/lang_probe.py --checkpoint <ckpt>/pretrained_model --in-scene
    $PY scripts/analysis/lang_probe.py --checkpoint <ckpt>/pretrained_model --compare lerobot/MolmoAct2-BimanualYAM-LeRobot

`--in-scene` restricts to objects visible in every probed frame; without it,
naming an absent object counts as a failure when it is really correct behaviour.

**3. Is it jerky?** Compares chunks from identical frames, with the human demo
as the reference.

    $PY scripts/analysis/compare_action_noise.py \
        --checkpoints lerobot/MolmoAct2-BimanualYAM-LeRobot <ckpt>/pretrained_model \
        --episodes 0 5 12 --frames 2

**4. Does it lunge toward the stow pose?** Dumps whole 30-step chunks — the
snap is invisible if you only look at step 0, because the robot executes all 30
open-loop.

    $PY scripts/analysis/inspect_policy_chunks.py --checkpoint <ckpt>/pretrained_model --episode 5

**5. Does the sim look different enough to change its behaviour?** Same state
and task, real images vs rendered. Needs the sim server running.

    $PY scripts/analysis/compare_real_vs_sim_images.py --checkpoint <ckpt>/pretrained_model --port 8081

## …work out why the loop is slow

Per-stage timing of a real tick, separating inference from everything else.
Takes all `lerobot-rollout` flags plus `--ticks` / `--warmup`.

    $PY scripts/analysis/profile_rollout.py \
        --policy.path=<ckpt>/pretrained_model \
        --robot.type=yam_ultra_sim --robot.port=8081 \
        --task="put the marker into the cardboard box" \
        --duration=0 --ticks 70

## …augment a dataset's instructions

Groups task strings by the object they name, generates phrasings per object,
and rotates each frame's `task_index` among them. Writes a new dataset with the
videos hardlinked, so it costs megabytes rather than gigabytes.

    # See the grouping and phrasings without writing anything
    $PY scripts/augment_task_paraphrases.py --src <dataset> --dst /tmp/x --dry-run

    $PY scripts/augment_task_paraphrases.py \
        --src ~/.cache/huggingface/lerobot/minhphd/put-box \
        --dst ~/.cache/huggingface/lerobot/minhphd/put-box-aug

Check `OBJECT_OF` in that script first — it decides which strings name the same
physical object, and merging two different objects teaches the model they are
interchangeable.

## …use Isaac

    ./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081
    ./scripts/isaac_python.sh my_script.py
    ./scripts/isaac_python.sh                    # REPL

    ./scripts/install_isaac_kernel.sh            # once
    ./scripts/start_isaac_jupyter.sh

Isaac ships its own Python 3.12 with **no lerobot**, which is why the sim runs
as a separate server process the rollout talks to over HTTP.

---

# Gotchas

**`--rename_map`** exists because checkpoints trained via `--policy.path`
expect `top/left/right` while the dataset records `top/left_wrist/right_wrist`.
A checkpoint whose `input_features` already say `left_wrist` must **not** get
it — that is the `RENAME` block in `rollout.sh`, selected by `--policy`.

**Two GPU-heavy processes will not both fit.** A rollout (~13 GB) plus the
Isaac server (~4 GB) plus a diagnostic (~13–27 GB) exceeds 32 GB. Finish one
before starting the next, or pass `--dtype bfloat16` where the script offers it.
