"""Dump whole predicted ACTION CHUNKS, unclamped, over a recorded episode.

Chases a specific observed failure: the arm occasionally snapping fast toward
the stow/zero pose mid-rollout. If the policy itself emits a chunk that heads
to zero, the follower's Δq clamp only slows that snap down — it cannot stop
it, because the clamp bounds per-tick step size, not where the trajectory is
going.

Why whole chunks and not ``select_action``'s return value: that returns only
step 0 of 30, but the robot executes all 30 open-loop. A chunk can start
perfectly reasonable and collapse toward zero by step 20 — which is exactly
what an occasional snap-to-stow would look like, and is invisible if you only
inspect step 0.

Nothing here goes near the follower, so no clamp, no gripper flip, no RPC —
this is the policy's raw intent, on recorded training frames.

Reads as:

  ``|chunk[k]|`` shrinking steadily toward 0 across k
      the policy is driving the arm home. That is the snap.

  ``|chunk[k]|`` staying near ``|state|``
      chunk is well behaved; the snap comes from somewhere else
      (check for a stray park() -- follower.park logs its caller).

    python scripts/inspect_policy_chunks.py --episode 5 --frames 12
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

ARM = list(range(6)) + list(range(7, 13))   # both arms, excluding grippers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="MolmoAct2/008000/pretrained_model")
    ap.add_argument("--repo-id", default="minhphd/put-box")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episode", type=int, default=5)
    ap.add_argument("--frames", type=int, default=12,
                    help="how many points through the episode to predict a chunk at")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--snap-jump", type=float, default=0.3,
                    help="flag a chunk whose FIRST action is this much closer to "
                         "zero (in joint-space norm) than the arm's actual pose")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.checkpoint, dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    n_steps = int(policy.config.n_action_steps)

    lo = ds.meta.episodes[args.episode]["dataset_from_index"]
    hi = ds.meta.episodes[args.episode]["dataset_to_index"]
    idxs = np.linspace(lo, hi - 1, args.frames).astype(int)
    print(f"episode {args.episode}: frames {lo}..{hi-1}, chunk={n_steps} steps\n")
    print(f"{'frame':>7} {'|state|':>8} {'|chunk[0]|':>11} {'jump->0':>9} "
          f"{'|chunk[-1]|':>12} {'max step':>9}  verdict")

    flagged = 0
    for i in idxs:
        item = ds[int(i)]
        batch = {k: (v.unsqueeze(0).to(args.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in item.items()}
        batch["task"] = item["task"] if isinstance(item["task"], str) else item["task"][0]

        # Drain a full chunk through the SAME path the robot uses: the first
        # select_action runs the model and fills the queue, the rest pop from
        # it. So this is exactly the sequence the arm would execute.
        policy.reset()
        chunk = []
        with torch.inference_mode():
            for _ in range(n_steps):
                a = policy.select_action(pre(batch), inference_action_mode="continuous")
                chunk.append(post(a).squeeze(0).float().cpu().numpy())
        chunk = np.stack(chunk)                      # (n_steps, 14)

        state = item["observation.state"].numpy()
        d = np.linalg.norm(chunk[:, ARM], axis=1)    # distance from zero per step
        d_state = float(np.linalg.norm(state[ARM]))
        max_step = float(np.abs(np.diff(chunk[:, ARM], axis=0)).max()) if n_steps > 1 else 0.0

        # The snap is the discontinuity between where the arm IS and where the
        # chunk STARTS — not drift inside the chunk (which is smooth). Each new
        # chunk yanks the arm toward zero by this much.
        jump = d_state - d[0]
        snapping = jump > args.snap_jump
        verdict = "<-- SNAPS TOWARD STOW" if snapping else ""
        flagged += bool(snapping)
        print(f"{int(i):>7} {d_state:>8.3f} {d[0]:>11.3f} {jump:>+9.3f} "
              f"{d[-1]:>12.3f} {max_step:>9.4f}  {verdict}")

    print(f"\n  {flagged}/{len(idxs)} chunks OPEN with a jump toward zero "
          f"> {args.snap_jump:g} rad")
    if flagged:
        print("  -> the POLICY yanks the arm toward stow at each chunk boundary.")
        print("     Classic regression-to-the-mean from an underfit model: the")
        print("     further the arm is from the folded pose, the harder the pull")
        print("     (see how `jump->0` grows with |state|). The Delta-q clamp bounds")
        print("     how FAST it gets there, not whether it goes -- so this cannot be")
        print("     fixed follower-side. It is the same underfitting the validation")
        print("     script reports; retraining is the fix.")
    else:
        print("  -> chunks open near the arm's actual pose; look elsewhere for the")
        print("     snap (follower.park() logs its caller -- check for a stray call).")


if __name__ == "__main__":
    main()
