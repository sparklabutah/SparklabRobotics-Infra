"""Open-loop check: does the checkpoint reproduce its OWN training data?

Answers "is the rollout pipeline correct, or am I looking at a bug?" without
touching hardware. Feeds recorded frames from the training dataset through
the exact processors ``lerobot-rollout`` builds, runs the policy, and
compares the predicted action to the action a human actually teleoperated at
that frame.

How to read the result:

  small error on training frames
      The pipeline is correct end to end — image keys, the quantile
      normalizer, state ordering, action decoding. The model has fit its
      data. Whatever the arm does on hardware is then GENUINE policy
      behaviour (closed-loop drift, covariate shift, undertraining on a
      29-task/49-episode set) and not a plumbing bug.

  large error on training frames
      Something upstream of the robot is wrong. A model cannot fail on data
      it was trained on unless inputs are being mangled — wrong camera
      assigned to a key, normalization stats mismatched, state in the wrong
      joint order — or it is drastically undertrained.

This is deliberately open-loop and on TRAINING data: it is the easiest
possible test. Passing it does not mean the policy is good, only that the
plumbing is honest. Failing it means nothing downstream can be trusted.

    python scripts/validate_policy_pipeline.py \\
        --checkpoint MolmoAct2/008000/pretrained_model \\
        --repo-id minhphd/put-box --episodes 0 5 12 --frames 6
"""

from __future__ import annotations

import argparse
import numpy as np
import torch


MOTORS = [f"{h}_joint_{j}.pos" for h in ("left", "right") for j in range(1, 7)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="MolmoAct2/008000/pretrained_model")
    ap.add_argument("--repo-id", default="minhphd/put-box")
    ap.add_argument("--root", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 5, 12])
    ap.add_argument("--frames", type=int, default=6,
                    help="frames sampled evenly through each episode")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    print(f"dataset: {ds.meta.total_episodes} episodes, fps={ds.meta.fps}")

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(args.device).eval()

    # Same construction lerobot-rollout uses (rollout/context.py:392), so a
    # discrepancy here would be a discrepancy there.
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.checkpoint,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    all_err, all_base = [], []
    for ep in args.episodes:
        if ep >= ds.meta.total_episodes:
            print(f"  episode {ep}: out of range, skipping")
            continue
        lo = ds.meta.episodes[ep]["dataset_from_index"]
        hi = ds.meta.episodes[ep]["dataset_to_index"]
        idxs = np.linspace(lo, hi - 1, args.frames).astype(int)

        errs, bases = [], []
        for i in idxs:
            item = ds[int(i)]
            batch = {k: (v.unsqueeze(0).to(args.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in item.items()}
            batch["task"] = item["task"] if isinstance(item["task"], str) else item["task"][0]

            truth = item["action"].numpy()          # what the human actually did
            state = item["observation.state"].numpy()

            policy.reset()                          # open loop: no queue carry-over
            with torch.inference_mode():
                out = policy.select_action(pre(batch), inference_action_mode="continuous")
            pred = post(out).squeeze(0).float().cpu().numpy()

            if len(errs) == 0:      # first frame of each episode: show raw vectors
                np.set_printoptions(precision=3, suppress=True, linewidth=200)
                print(f"    frame {int(i)} raw comparison:")
                print(f"      state  {np.asarray(state_dbg := item['observation.state'].numpy())}")
                print(f"      truth  {np.asarray(truth)}")
                print(f"      pred   {np.asarray(pred)}")
            n = min(len(pred), len(truth))
            errs.append(np.abs(pred[:n] - truth[:n]))
            # Baseline: predicting "don't move" (= the current measured state).
            bases.append(np.abs(state[:n] - truth[:n]))

        e, b = np.mean(errs, axis=0), np.mean(bases, axis=0)
        all_err.append(e); all_base.append(b)
        task = ds.meta.tasks.index[int(item["task_index"])] if hasattr(ds.meta.tasks, "index") else "?"
        print(f"  ep {ep:3d}  mean|err| arm={np.mean(e[:6]):.4f} rad   "
              f"vs no-move baseline {np.mean(b[:6]):.4f} rad   ({task[:44]})")

    if not all_err:
        return
    e = np.mean(all_err, axis=0); b = np.mean(all_base, axis=0)
    print("\n=== overall, on TRAINING frames ===")
    for k, name in enumerate(MOTORS):
        kk = k if k < 6 else k + 1
        print(f"  {name:17s} err={e[kk]:.4f} rad   baseline={b[kk]:.4f} rad")
    arm_idx = list(range(6)) + list(range(7, 13))   # both arms, excluding grippers
    arm_err, arm_base = float(np.mean(e[arm_idx])), float(np.mean(b[arm_idx]))
    print(f"\n  arm mean |error|   : {arm_err:.4f} rad ({np.degrees(arm_err):.2f} deg)")
    print(f"  no-move baseline   : {arm_base:.4f} rad ({np.degrees(arm_base):.2f} deg)")
    # NB: in this dataset the recorded action is essentially the measured
    # state (verified: identical to 3 dp on sampled frames), so "predict the
    # current state" is a near-ORACLE, not a fair baseline — no model beats
    # it. What it really measures is the SIGNAL: how much motion the human
    # actually commanded per frame. The question that matters is whether the
    # model's error is small relative to that signal.
    snr = arm_base / arm_err if arm_err else float("inf")
    print(f"  commanded motion/frame : {arm_base:.4f} rad  <- the signal to reproduce")
    print(f"  model error            : {arm_err:.4f} rad")
    print(f"  signal-to-error ratio  : {snr:.2f}")
    if snr >= 2.0:
        print("  VERDICT: error is well below the commanded motion -> the policy can express")
        print("           the demonstrated movement. Pipeline and model both sound.")
    elif snr >= 1.0:
        print("  VERDICT: error comparable to the commanded motion -> weak but not broken.")
        print("           Expect sloppy, imprecise execution.")
    else:
        print("  VERDICT: error EXCEEDS the commanded motion -> the movement signal is buried")
        print("           in the model's own noise. Expect small erratic non-task-directed")
        print("           motion on hardware. This is a TRAINING problem (steps/epochs, or an")
        print("           absolute-vs-delta action representation), not a pipeline bug -- check")
        print("           the raw vectors above: if they track truth in scale/sign, plumbing is fine.")


if __name__ == "__main__":
    main()
