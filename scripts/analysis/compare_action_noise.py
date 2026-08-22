"""Is one checkpoint's action chunk noisier than another's?

Feeds the SAME recorded frames to two checkpoints and measures how jittery the
30-step chunk each one predicts is, with the human's own teleoperated actions
over the same horizon as the reference for "how smooth should this be".

    python scripts/compare_action_noise.py \\
        --checkpoints lerobot/MolmoAct2-BimanualYAM-LeRobot \\
                      ../SparkRobot/MolmoAct2/008000/pretrained_model \\
        --episodes 0 5 12 --frames 4

WHAT IS MEASURED
Noise here means high-frequency content along the chunk's time axis, not
distance from the truth -- a policy can be smooth and wrong, or accurate and
shaky, and those need different fixes.

  step  : mean |a[t+1] - a[t]|          how fast it moves
  jerk  : mean |a[t+1] - 2a[t] + a[t-1]| how much it CHANGES direction
  ratio : jerk / step                    direction changes per unit motion

``ratio`` is the one to read. Raw jerk scales with speed, so a policy that
simply moves faster looks jerkier; dividing by step size removes that. A
trajectory that accelerates smoothly has a small ratio whatever its speed; one
that dithers back and forth has a large one regardless.

Chunks are compared open-loop from identical inputs, so any difference is the
model, not the sim, the loop rate or the clamp.
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import torch


# The hardware follower's per-tick allowance at 30 Hz:
#   velocity [4,4,4,6,6,6] rad/s * 1/30  = [.133,.133,.133,.200,.200,.200]
#   intersected with max_relative_target = 0.15
CAPS_30HZ = np.array([0.1333, 0.1333, 0.1333, 0.15, 0.15, 0.15])


def chunk_metrics(a: np.ndarray) -> dict:
    """a: [T, A] action chunk. Returns per-trajectory smoothness numbers."""
    if a.ndim != 2 or a.shape[0] < 3:
        return {"step": float("nan"), "jerk": float("nan"), "ratio": float("nan"),
                "flips": float("nan"), "p_clamped": float("nan"),
                "max_step_deg": float("nan"), "dither_deg": float("nan")}
    d1 = np.diff(a, axis=0)
    d2 = np.diff(a, n=2, axis=0)
    step = float(np.abs(d1).mean())
    jerk = float(np.abs(d2).mean())
    # Sign flips: how often each joint reverses direction along the chunk. A
    # direct count of dithering, independent of magnitude entirely.
    flips = float((np.diff(np.sign(d1), axis=0) != 0).mean())

    # Would the hardware clamp ever fire on this trajectory? Only the 6 arm
    # joints per hand are capped; the gripper is exempt. Columns are
    # [left 0..5, left_grip 6, right 7..12, right_grip 13].
    arm = np.concatenate([np.abs(d1[:, 0:6]), np.abs(d1[:, 7:13])], axis=1)
    caps = np.concatenate([CAPS_30HZ, CAPS_30HZ])
    p_clamped = float((arm > caps).mean())
    max_step_deg = float(np.degrees(arm.max()))

    # Amplitude of the direction reversals themselves: how big is the wobble
    # the arm would actually be asked to perform? This is what decides whether
    # real hardware can even express it.
    rev = np.abs(d2[:, list(range(6)) + list(range(7, 13))])
    dither_deg = float(np.degrees(rev.mean()))

    return {"step": step, "jerk": jerk,
            "ratio": jerk / step if step > 1e-9 else float("nan"),
            "flips": flips, "p_clamped": p_clamped,
            "max_step_deg": max_step_deg, "dither_deg": dither_deg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--repo-id", default="minhphd/put-box")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 5, 12])
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    print(f"dataset: {ds.meta.total_episodes} episodes, fps={ds.meta.fps}")

    # Pick the frames ONCE so every checkpoint sees identical inputs.
    picks: list[int] = []
    for ep in args.episodes:
        if ep >= ds.meta.total_episodes:
            continue
        lo = ds.meta.episodes[ep]["dataset_from_index"]
        hi = ds.meta.episodes[ep]["dataset_to_index"]
        picks += np.linspace(lo, hi - 1, args.frames).astype(int).tolist()
    print(f"comparing on {len(picks)} frames from episodes {args.episodes}\n")

    results: dict[str, list[dict]] = {}

    for ckpt in args.checkpoints:
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = args.device
        policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
        policy.to(args.device).eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=ckpt,
            dataset_stats=ds.meta.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        want = [k for k in cfg.input_features if "image" in k]
        horizon = int(getattr(cfg, "chunk_size", 30))

        per_frame = []
        for i in picks:
            item = ds[int(i)]
            batch = {k: (v.unsqueeze(0).to(args.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in item.items()}
            batch["task"] = item["task"] if isinstance(item["task"], str) else item["task"][0]

            # Some checkpoints were trained with .left/.right rather than
            # .left_wrist/.right_wrist (that is what rollout_pretrained.sh's
            # --rename_map exists for). Alias rather than fail, so the base and
            # the finetune can be compared on one command line.
            for k in want:
                if k not in batch:
                    alt = (k.replace(".left", ".left_wrist")
                            .replace(".right", ".right_wrist"))
                    if alt in batch:
                        batch[k] = batch[alt]

            policy.reset()
            with torch.inference_mode():
                out = policy.predict_action_chunk(pre(batch),
                                                  inference_action_mode="continuous")
            chunk = post(out).squeeze(0).float().cpu().numpy()[:horizon]
            per_frame.append(chunk_metrics(chunk))

        results[ckpt] = per_frame

        # One MolmoAct2 fills most of a 32 GB card, so the next checkpoint can
        # only load if this one is genuinely gone. `del` alone is not enough:
        # the policy's submodules reference each other, so the tensors stay
        # reachable until a cycle collection runs, and empty_cache() then frees
        # nothing. Move to CPU first so the release does not depend on when the
        # collector happens to fire.
        policy.to("cpu")
        del policy, pre, post, cfg
        gc.collect()
        torch.cuda.empty_cache()

    # The human's own actions over the same horizon: the reference smoothness.
    truth = []
    for i in picks:
        lo, hi = 0, len(ds)
        seg = np.stack([ds[min(int(i) + t, hi - 1)]["action"].numpy()
                        for t in range(30)])
        truth.append(chunk_metrics(seg))
    results["HUMAN DEMO (reference)"] = truth

    print(f"{'checkpoint':<46}{'step':>8}{'ratio':>7}{'flips':>7}"
          f"{'%clamp':>8}{'maxstep':>9}{'dither':>8}")
    for name, rows in results.items():
        m = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
        label = name if len(name) <= 44 else "..." + name[-41:]
        print(f"{label:<46}{m['step']:8.4f}{m['ratio']:7.2f}{m['flips']:7.2f}"
              f"{100*m['p_clamped']:7.1f}%{m['max_step_deg']:9.2f}"
              f"{m['dither_deg']:8.3f}")
    print()
    print("%clamp  = share of per-tick steps that EXCEED the hardware cap")
    print("           (0% means the clamp never fires -- it is not what")
    print("            makes the arm smooth)")
    print("maxstep = largest single-tick move in the chunk, degrees")
    print("           (cap is 7.6 deg for joints 1-3, 8.6 deg for 4-6)")
    print("dither  = mean size of the direction reversals, degrees")

    print("\nratio = jerk/step: direction changes per unit of motion.")
    print("flips = fraction of steps where a joint reversed direction.")
    print("Compare each model against HUMAN DEMO, not against zero -- real")
    print("teleop is not perfectly smooth either.")


if __name__ == "__main__":
    main()
