#!/usr/bin/env python3
"""
Does the policy read the instruction, or does it ignore it?

Feeds ONE observation to the policy several times, changing only the task
string, and measures how much the predicted action chunk moves. That isolates
language conditioning from everything else -- no robot, no control loop, no
timing.

Two numbers are computed per frame:

  cross-instruction std   how much the chunk changes when the TARGET changes
  paraphrase std          how much it changes when the wording changes but the
                          target does not

The second is the control. A policy that reads language should show
cross-instruction >> paraphrase: swapping "marker" for "screwdriver" should
move the trajectory, swapping "put the marker" for "pick up the marker and
put it" should not. If the two are similar, the model is responding to
surface text rather than to meaning. If both are ~0, it is ignoring the text
channel entirely and reaching for whatever it always reaches for.

    python lang_probe.py --checkpoint <path>/pretrained_model
    python lang_probe.py --checkpoint <path> --compare <other-checkpoint>

NOTE ON KEYS: a checkpoint trained via --policy.path expects the LeRobot
checkpoint's camera names (top / left / right) while the dataset uses
top / left_wrist / right_wrist. --rename is applied automatically when the
policy's input_features disagree with the dataset's.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

# Same target, different wording -- these must NOT move the trajectory much.
PARAPHRASES = [
    "put the marker into the cardboard box",
    "put the black marker into the cardboard box",
    "put the marker into the box",
    "pick up the marker and put it into the cardboard box",
]

# Different targets -- these SHOULD move the trajectory.
TARGETS = [
    "put the marker into the cardboard box",
    "put the screwdriver into the cardboard box",
    "put the red e-stop button into the cardboard box",
    "put the red toolbox into the cardboard box",
    "put the measuring tape into the cardboard box",
    "put the electrical tape into the cardboard box",
    "put the red scissor into the cardboard box",
]

# Only the objects VISIBLE in every probed frame (--in-scene).
#
# Checked by eye against the top camera at the six frames this probes: the
# e-stop, screwdriver, electrical tape, marker and box are on the table in all
# of them, while the red toolbox, scissors and measuring tape appear in only
# two or three. Naming an absent object and getting no trajectory change is
# CORRECT behaviour, not a grounding failure, so including those biases the
# result toward "ignores language". This list removes that bias at the cost of
# a smaller spread (4 targets instead of 7).
TARGETS_IN_SCENE = [
    "put the marker into the cardboard box",
    "put the screwdriver into the cardboard box",
    "put the electrical tape into the cardboard box",
    "put the red e-stop button into the cardboard box",
]


def load(checkpoint, device="cuda"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    # Dispatch on the checkpoint's own declared type instead of hardcoding one
    # policy class, so the same probe runs against molmoact2, pi05, or anything
    # else registered. The question -- does the instruction change the action
    # chunk more than resampling does -- is policy-agnostic.
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.device = device
    pol = get_policy_class(cfg.type).from_pretrained(str(checkpoint), config=cfg)
    pol.eval()
    # The device_processor step is SAVED in policy_preprocessor.json, so it
    # carries whatever device the checkpoint was written with. The finetunes
    # happen to say cuda; the released base says cpu, which leaves input_ids
    # on the CPU while the model sits on the GPU:
    #   "index is on cpu, different from other tensors on cuda:0"
    # Overriding it here makes the probe checkpoint-agnostic.
    pre, post = make_pre_post_processors(
        pol.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}})
    return pol, pre, post


def chunk_for(pol, pre, post, sample, task, rename, device, seed):
    batch = {}
    for k, v in sample.items():
        k2 = rename.get(k, k)
        batch[k2] = v.unsqueeze(0) if torch.is_tensor(v) else [v]
    batch["task"] = [task]
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = post(pol.predict_action_chunk(pre(batch)))
    return out.squeeze(0).float().cpu().numpy()


def probe(checkpoint, ds, frames, n_seeds, device, targets=None):
    targets = targets or TARGETS
    pol, pre, post = load(checkpoint, device)
    pol.to(device)

    want = set(pol.config.input_features)
    have = set(ds.features)
    # Map this dataset's camera names onto whatever the checkpoint expects.
    # The pairs are (dataset name, checkpoint name) and are per-family, not
    # guessable: a LeRobot-format MolmoAct2 wants top/left/right, while pi05
    # inherits openpi's base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb --
    # where "base" is the external scene view, i.e. our top camera.
    CAMERA_ALIASES = [
        ("left_wrist", "left"), ("right_wrist", "right"),          # molmoact2
        ("top", "base_0_rgb"),                                     # pi05 (openpi)
        ("left_wrist", "left_wrist_0_rgb"),
        ("right_wrist", "right_wrist_0_rgb"),
    ]
    rename = {}
    for a, b in CAMERA_ALIASES:
        src, dst = f"observation.images.{a}", f"observation.images.{b}"
        if src in have and dst in want and dst not in have:
            rename[src] = dst
    if rename:
        print(f"  renaming keys: {rename}")
    # A camera the checkpoint wants but nothing supplies means the probe would
    # be feeding it blanks, and every measurement after that is meaningless.
    unmet = {k for k in want if "image" in k} - set(rename.values()) - have
    if unmet:
        raise SystemExit(
            f"checkpoint expects image keys with no source in this dataset: "
            f"{sorted(unmet)}\n  dataset has: "
            f"{sorted(k for k in have if 'image' in k)}\n"
            f"  add the pairing to CAMERA_ALIASES")

    rows = []
    for fi in frames:
        sample = ds[int(fi)]

        # Sampling noise floor: same instruction, different seeds. Anything
        # below this is indistinguishable from stochastic sampling.
        base = np.stack([chunk_for(pol, pre, post, sample, targets[0],
                                   rename, device, s) for s in range(n_seeds)])
        noise = float(base.std(axis=0).mean())

        # Hold the seed fixed so the ONLY thing varying is the text.
        tgt = np.stack([chunk_for(pol, pre, post, sample, t, rename, device, 0)
                        for t in targets])
        par = np.stack([chunk_for(pol, pre, post, sample, t, rename, device, 0)
                        for t in PARAPHRASES])

        cross = float(tgt.std(axis=0).mean())
        para = float(par.std(axis=0).mean())
        within = float(tgt.std(axis=1).mean())     # motion scale, for context

        rows.append(dict(frame=int(fi), noise=noise, cross=cross,
                         para=para, within=within))
    return rows


def report(name, rows):
    print(f"\n=== {name}")
    print(f"{'frame':>7} {'noise':>9} {'cross':>9} {'para':>9} "
          f"{'cross/noise':>12} {'cross/para':>11} {'cross/within':>13}")
    for r in rows:
        print(f"{r['frame']:>7} {r['noise']:>9.4f} {r['cross']:>9.4f} "
              f"{r['para']:>9.4f} {r['cross']/(r['noise']+1e-9):>12.2f} "
              f"{r['cross']/(r['para']+1e-9):>11.2f} "
              f"{r['cross']/(r['within']+1e-9):>13.4f}")
    c = np.median([r["cross"] for r in rows])
    n = np.median([r["noise"] for r in rows])
    p = np.median([r["para"] for r in rows])
    print(f"\n  median cross/noise = {c/(n+1e-9):.2f}   "
          f"cross/paraphrase = {c/(p+1e-9):.2f}")
    if c / (n + 1e-9) < 1.5:
        print("  -> instruction moves the chunk no more than resampling does.")
        print("     The policy is NOT conditioning on the target object.")
    elif c / (p + 1e-9) < 1.5:
        print("  -> target swaps and paraphrases move it about equally:")
        print("     responding to surface text, not to which object is named.")
    else:
        print("  -> target identity changes the trajectory well beyond both")
        print("     sampling noise and paraphrase. Language IS conditioning.")


def pick_frames(ds, n_frames, frac):
    """``n_frames`` global indices, each ``frac`` of the way into its episode.

    Replaces ``linspace(len(ds)*0.15, len(ds)*0.85)``, which indexed the whole
    CONCATENATED dataset rather than an episode. That version's comment claimed
    to avoid the parked arm at t=0, but nothing enforced it -- where an index
    landed inside its episode was whatever the stride happened to alias to. On
    this dataset it produced 59-93% into each episode, i.e. always the back
    half, purely by luck of 754-frame episodes against a ~5170-frame stride.

    That mattered: in a "pick up X and put it in the box" episode the back half
    is AFTER the grasp, where the remaining trajectory is the object-agnostic
    common suffix (carry to box, release). Probing only there suppresses exactly
    the cross-instruction variance this script exists to measure, biasing every
    result toward "language is dead".

    Choosing the phase explicitly makes that a parameter instead of an accident:
    ~0.25 is the reach (object identity still matters), ~0.75 the transport.
    """
    eps = ds.meta.episodes
    starts = eps["dataset_from_index"]
    ends = eps["dataset_to_index"]
    n_eps = len(starts)
    # Spread the probes over distinct episodes so a single odd demonstration
    # cannot dominate; clamp inside the episode so frac=1.0 is still valid.
    chosen = np.unique(np.linspace(0, n_eps - 1, n_frames).astype(int))
    out = []
    for e in chosen:
        lo, hi = int(starts[e]), int(ends[e])
        out.append(min(lo + int((hi - lo) * frac), hi - 1))
    return np.array(out), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compare", default=None,
                    help="second checkpoint to run identically (e.g. the base)")
    ap.add_argument("--repo-id", default="minhphd/put-box")
    ap.add_argument("--root", default="~/.cache/huggingface/lerobot/minhphd/put-box")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--in-scene", action="store_true",
                    help="restrict targets to objects visible in every probed "
                         "frame; naming an absent object cannot move the "
                         "trajectory, which biases the result")
    ap.add_argument("--episode-frac", type=float, nargs="+", default=[0.25, 0.75],
                    help="how far into each episode to probe, 0..1. Give more "
                         "than one to compare phases: ~0.25 is the reach, where "
                         "which object was named still decides the trajectory; "
                         "~0.75 is the transport, where every instruction "
                         "produces the same carry-to-box suffix. A model that "
                         "grounds at 0.25 and not at 0.75 is behaving correctly")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=str(Path(args.root).expanduser()))
    print(f"dataset: {ds.meta.total_episodes} eps, {len(ds)} frames")

    targets = TARGETS_IN_SCENE if args.in_scene else TARGETS
    print(f'targets ({len(targets)}): ' + '; '.join(targets))

    for frac in args.episode_frac:
        frames, eps = pick_frames(ds, args.n_frames, frac)
        print(f"\n########## {frac:.0%} into each episode")
        print(f"episodes {eps.tolist()} -> frames {frames.tolist()}")
        report(f"{args.checkpoint}  @{frac:.0%}",
               probe(args.checkpoint, ds, frames, args.n_seeds, args.device, targets))
        if args.compare:
            report(f"{args.compare}  @{frac:.0%}",
                   probe(args.compare, ds, frames, args.n_seeds, args.device, targets))

    if len(args.episode_frac) > 1:
        print("\nCompare the phases before concluding anything. Grounding that is")
        print("strong early and weak late is CORRECT: once the object is grasped,")
        print("every instruction shares the same carry-to-box suffix. Weak at both")
        print("is the result that actually indicts the model.")
    if args.compare:
        print("\nIf the base grounds and yours does not, fine-tuning cost you")
        print("language conditioning -- expected when one class (marker, 11 of")
        print("49) dominates and the action head starts predicting the mode.")


if __name__ == "__main__":
    main()