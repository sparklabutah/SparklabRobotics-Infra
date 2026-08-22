"""Does the sim's appearance actually change what the policy does?

The policy was trained on real camera frames and is being evaluated in sim.
Appearance statistics already show a gap (sim is brighter, wrist views carry a
third of the edge density). Statistics are not behaviour, so this measures the
only thing that matters: feed the SAME state and the SAME task, vary only the
images, and see whether the predicted action chunk moves.

    PYTHONPATH=src python scripts/compare_real_vs_sim_images.py \\
        --checkpoint ../SparkRobot/MolmoAct2/lerobot_training/last/pretrained_model \\
        --port 8081 --frames 6

THE CONTROL MATTERS MORE THAN THE COMPARISON

MolmoAct2 samples its actions (flow matching), so two runs on IDENTICAL input
already differ. Reporting real-vs-sim alone would be meaningless -- it has to
be read against real-vs-real resampling noise on the same frames. So each
frame is predicted three times:

    A1 = policy(real images)
    A2 = policy(real images)     <- same input, different sample
    B  = policy(sim images)

and the number that matters is  |A1-B| / |A1-A2|.  At 1.0 the sim is
indistinguishable from resampling noise; well above 1.0 and the domain gap is
steering the policy.

WHAT THIS DOES *NOT* ISOLATE

The sim scene is not a replica of the frame being compared against: the props
sit where scene.usda puts them, not where they were in that episode. So a
difference here is "sim frame vs real frame", which bundles rendering fidelity
WITH scene-content mismatch. That is the right question for "can I trust a sim
rollout", but it is the wrong question for "is my renderer good enough", and
the two should not be confused.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def sim_images(url: str, state_vec, joint_names, timeout=60.0):
    """Drive the sim to ``state_vec`` and return its three camera frames."""
    import base64
    import json
    import urllib.request

    import cv2

    action = {n: float(v) for n, v in zip(joint_names, state_vec)}
    req = urllib.request.Request(
        url + "/step", data=json.dumps({"action": action}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    if "error" in body:
        raise RuntimeError(body["error"])
    out = {}
    for k, b64 in body["images"].items():
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out[k] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", default="minhphd/put-box")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 5, 12])
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}"

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    ds = LeRobotDataset(repo_id=args.repo_id)
    names = ds.meta.features["observation.state"]["names"]
    print(f"dataset: {ds.meta.total_episodes} episodes, state dim {len(names)}")

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    if hasattr(cfg, "model_dtype"):
        cfg.model_dtype = args.dtype      # bf16 fits alongside Isaac; fp32 does not
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.checkpoint,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}})

    img_keys = [k for k in cfg.input_features if "image" in k]
    print(f"policy image inputs: {img_keys}")
    # Map a policy image key onto the sim server's camera name.
    def cam_of(key):
        tail = key.split(".")[-1]
        return {"left": "left_wrist", "right": "right_wrist"}.get(tail, tail)

    picks = []
    for ep in args.episodes:
        if ep >= ds.meta.total_episodes:
            continue
        lo = ds.meta.episodes[ep]["dataset_from_index"]
        hi = ds.meta.episodes[ep]["dataset_to_index"]
        picks += np.linspace(lo, hi - 1, args.frames).astype(int).tolist()
    print(f"comparing on {len(picks)} frames from episodes {args.episodes}\n")

    def predict(batch):
        policy.reset()
        with torch.inference_mode():
            out = policy.predict_action_chunk(pre(batch),
                                              inference_action_mode="continuous")
        return post(out).squeeze(0).float().cpu().numpy()

    rows = []
    for i in picks:
        item = ds[int(i)]
        base = {k: (v.unsqueeze(0).to(args.device) if isinstance(v, torch.Tensor) else v)
                for k, v in item.items()}
        base["task"] = item["task"] if isinstance(item["task"], str) else item["task"][0]
        for k in img_keys:
            if k not in base:
                alt = k.replace(".left", ".left_wrist").replace(".right", ".right_wrist")
                if alt in base:
                    base[k] = base[alt]

        # Sim frames at this frame's exact joint state.
        state = item["observation.state"].numpy().tolist()
        sims = sim_images(url, state, names)

        sim_batch = dict(base)
        for k in img_keys:
            arr = sims[cam_of(k)].astype(np.float32) / 255.0     # HWC -> CHW, [0,1]
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)
            ref = base[k]
            if t.shape[-2:] != ref.shape[-2:]:
                t = torch.nn.functional.interpolate(t, size=ref.shape[-2:],
                                                    mode="bilinear", align_corners=False)
            sim_batch[k] = t.to(ref.dtype)

        a1, a2, b = predict(base), predict(base), predict(sim_batch)
        noise = np.abs(a1 - a2).mean()
        gap = np.abs(a1 - b).mean()
        rows.append((noise, gap, np.abs(a1 - a2), np.abs(a1 - b)))
        print(f"  frame {i:>6}   real-vs-real {noise:.4f}   real-vs-sim {gap:.4f}"
              f"   ratio {gap / noise if noise else float('nan'):5.2f}")

    n = np.mean([r[0] for r in rows])
    g = np.mean([r[1] for r in rows])
    print("\n" + "=" * 66)
    print(f"real-vs-real (resampling noise)  {n:.4f} rad   <- the floor")
    print(f"real-vs-sim  (domain gap)        {g:.4f} rad")
    print(f"ratio                            {g / n:.2f}x")
    if g / n < 1.3:
        print("\n-> sim frames move the policy no more than resampling does.")
        print("   The appearance gap is cosmetic for THIS policy.")
    else:
        print(f"\n-> sim frames move the policy {g / n:.1f}x more than resampling.")
        print("   The appearance gap is steering it; sim results will not")
        print("   transfer, and closing the visual gap matters more than speed.")

    per_noise = np.mean([r[2] for r in rows], axis=0).mean(axis=0)
    per_gap = np.mean([r[3] for r in rows], axis=0).mean(axis=0)
    print(f"\n{'joint':<20}{'noise':>10}{'gap':>10}{'ratio':>8}")
    action_names = ds.meta.features["action"]["names"]
    for j, nm in enumerate(action_names):
        r = per_gap[j] / per_noise[j] if per_noise[j] else float("nan")
        print(f"{nm:<20}{per_noise[j]:10.4f}{per_gap[j]:10.4f}{r:8.2f}")


if __name__ == "__main__":
    main()
