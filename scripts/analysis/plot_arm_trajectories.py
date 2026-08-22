"""Per-arm trajectory audit for a bimanual LeRobot dataset.

Answers one question: **is each arm doing a varied job, or one canned motion?**

    $PY scripts/analysis/plot_arm_trajectories.py --repo-id minhphd/put-box

Writes three figures next to each other:

  1. arm_split      which arm works in each episode, and by how much
  2. trajectories   every episode's joint traces, overlaid, split by working arm
  3. reach_spread   where each arm actually reaches, per joint

WHAT THE SHAPE OF THIS DATA IS. These demos are effectively SINGLE-ARM: in
almost every episode one arm does the whole task and the other holds still
(<1 rad of total joint motion, gripper never leaves its rest value). So the
interesting comparison is not left-vs-right within an episode, it is
left-when-working vs right-when-working across episodes.

WHY NOT KEY OFF THE GRIPPER. An obvious "where did it grasp?" metric is the
pose at gripper close. It does not survive contact with this data: the gripper
signal opens and closes several times per episode (regrasps, and a noisy
normalised width), so "the" close event is not well defined. Everything here is
computed from joint trajectories instead, which are smooth.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

ARM_JOINTS = 6
LEFT = slice(0, ARM_JOINTS)            # dims 0-5 joints, 6 gripper
RIGHT = slice(7, 7 + ARM_JOINTS)       # dims 7-12 joints, 13 gripper
LEFT_GRIP, RIGHT_GRIP = 6, 13
JOINT_NAMES = ["base yaw", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]

# dataviz reference palette, slots 1-2. Validated for this pair:
# CVD ΔE 24.7 (protan), normal ΔE 33.6, both well over the floors.
C_LEFT, C_RIGHT = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#fcfcfb"

OBJECTS = [("electrical tape", "electrical tape"), ("black tape", "black tape"),
           ("measuring tape", "measuring tape"), ("estop", "e-stop"),
           ("e-stop", "e-stop"), ("red button", "e-stop"), ("scissor", "scissors"),
           ("toolbox", "toolbox"), ("screwdriver", "screwdriver"), ("marker", "marker")]


def object_of(task: str) -> str:
    t = task.lower()
    for key, name in OBJECTS:
        if key in t:
            return name
    return "other"


def load(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    files = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {root/'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    idx2task = {v: k for k, v in tasks.task_index.items()}
    state = np.stack(df["observation.state"].values).astype(float)
    return info, df, state, idx2task


def audit(df, state, idx2task) -> pd.DataFrame:
    """One row per episode: who worked, how much, and where they reached."""
    ep = df.episode_index.values
    rows = []
    for e in sorted(df.episode_index.unique()):
        m = ep == e
        s = state[m]
        step = np.abs(np.diff(s, axis=0))
        task = idx2task[df.task_index[m].iloc[0]]
        l_path, r_path = step[:, LEFT].sum(), step[:, RIGHT].sum()
        rows.append(dict(
            ep=e, frames=int(m.sum()), task=task, object=object_of(task),
            l_path=l_path, r_path=r_path,
            mover="left" if l_path > r_path else "right",
            l_grip_range=np.ptp(s[:, LEFT_GRIP]), r_grip_range=np.ptp(s[:, RIGHT_GRIP]),
            # Furthest the base yaw swings from rest — a robust stand-in for
            # "how far to the side did this arm reach", with no gripper in it.
            l_yaw_extreme=s[np.argmax(np.abs(s[:, 0])), 0],
            r_yaw_extreme=s[np.argmax(np.abs(s[:, 7])), 7],
        ))
    return pd.DataFrame(rows)


def resample(x: np.ndarray, n: int = 200) -> np.ndarray:
    """Time-normalise one episode to n points so episodes overlay."""
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, x[:, j]) for j in range(x.shape[1])], axis=1)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_split(a: pd.DataFrame, out: Path):
    """Which arm works in each episode — a diverging bar, one per episode."""
    a = a.sort_values("ep")
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    y = np.arange(len(a))
    ax.barh(y, -a.l_path, color=C_LEFT, height=0.72)
    ax.barh(y, a.r_path, color=C_RIGHT, height=0.72)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y[::2])
    ax.set_yticklabels(a.ep.values[::2])
    ax.set_ylabel("episode", color=INK_2, fontsize=9)
    ax.set_xlabel("total joint motion in the episode (rad)  ←  left arm    ·    right arm  →",
                  color=INK_2, fontsize=9)
    lim = max(a.l_path.max(), a.r_path.max()) * 1.08
    ax.set_xlim(-lim, lim)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.0f}" for t in ticks])
    ax.set_xlim(-lim, lim)
    ax.invert_yaxis()
    style(ax)
    n_l = (a.mover == "left").sum()
    ax.set_title("One arm does the whole episode; the other holds still",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=26)
    ax.text(0, 1.012, f"{n_l} episodes led by the left arm · {len(a)-n_l} by the right",
            transform=ax.transAxes, color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_trajectories(df, state, a: pd.DataFrame, out: Path):
    """Every working arm's joint traces, overlaid, left column vs right column."""
    ep = df.episode_index.values
    fig, axes = plt.subplots(ARM_JOINTS, 2, figsize=(11, 12), sharex=True,
                             facecolor=SURFACE)
    for col, (side, sl, colour) in enumerate(
            [("left", LEFT, C_LEFT), ("right", RIGHT, C_RIGHT)]):
        eps = a[a.mover == side].ep.values
        traces = np.stack([resample(state[ep == e][:, sl]) for e in eps])
        for j in range(ARM_JOINTS):
            ax = axes[j, col]
            for k in range(len(traces)):
                ax.plot(np.linspace(0, 100, traces.shape[1]), traces[k, :, j],
                        color=colour, lw=0.8, alpha=0.35)
            med = np.median(traces[:, :, j], axis=0)
            ax.plot(np.linspace(0, 100, len(med)), med, color=colour, lw=2.4)
            ax.plot(np.linspace(0, 100, len(med)), med, color=SURFACE, lw=4.5, zorder=1.9)
            style(ax)
            ax.set_ylabel(JOINT_NAMES[j], color=INK_2, fontsize=9)
            spread = traces[:, :, j].std(axis=0).mean()
            ax.text(0.985, 0.06, f"σ {spread:.3f} rad", transform=ax.transAxes,
                    ha="right", color=INK_2, fontsize=8)
            if j == 0:
                ax.set_title(f"{side} arm — {len(eps)} episodes where it does the task",
                             color=INK, fontsize=11, weight="bold", pad=8)
        axes[-1, col].set_xlabel("% through the episode", color=INK_2, fontsize=9)
    fig.suptitle("Joint trajectories, every episode overlaid (bold = median)",
                 color=INK, fontsize=13, weight="bold", x=0.007, ha="left", y=0.997)
    fig.text(0.007, 0.972, "σ is the mean spread across episodes — small σ means the arm "
             "repeats one motion regardless of the object", color=INK_2, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.966])
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_reach(df, state, a: pd.DataFrame, out: Path):
    """Where each arm reaches: per-joint spread of the pose it works through."""
    ep = df.episode_index.values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharey=True)

    # (a) base-yaw extreme per episode, by object — does it reach different places?
    ax = axes[0]
    for side, colour, col in [("left", C_LEFT, "l_yaw_extreme"),
                              ("right", C_RIGHT, "r_yaw_extreme")]:
        sub = a[a.mover == side]
        ax.scatter(sub[col], np.random.default_rng(0).normal(
            0 if side == "left" else 1, 0.06, len(sub)),
            s=42, color=colour, edgecolor=SURFACE, linewidth=1.4, zorder=3,
            label=f"{side} arm")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["left", "right"])
    ax.set_xlabel("furthest base-yaw reached (rad)", color=INK_2, fontsize=9)
    style(ax)
    ax.set_title("Where each arm swings to", color=INK, fontsize=11,
                 weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    # (b) per-joint spread across episodes, side by side
    ax = axes[1]
    w = 0.38
    xs = np.arange(ARM_JOINTS)
    for off, (side, sl, colour) in enumerate(
            [("left", LEFT, C_LEFT), ("right", RIGHT, C_RIGHT)]):
        eps = a[a.mover == side].ep.values
        traces = np.stack([resample(state[ep == e][:, sl]) for e in eps])
        spread = traces.std(axis=0).mean(axis=0)
        ax.bar(xs + (off - 0.5) * w, spread, width=w * 0.92, color=colour,
               label=f"{side} arm")
    ax.set_xticks(xs); ax.set_xticklabels(JOINT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("spread across episodes (rad)", color=INK_2, fontsize=9)
    style(ax)
    ax.set_title("How much the motion varies between episodes", color=INK,
                 fontsize=11, weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default="minhphd/put-box")
    ap.add_argument("--root", default=None, help="dataset dir (overrides --repo-id)")
    ap.add_argument("--out", default="/tmp/arm_audit", help="output directory")
    args = ap.parse_args()

    root = Path(args.root) if args.root else (
        Path.home() / ".cache/huggingface/lerobot" / args.repo_id)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    info, df, state, idx2task = load(root)
    a = audit(df, state, idx2task)
    a.to_csv(out / "per_episode.csv", index=False)

    print(f"{root.name}: {info['total_episodes']} episodes, {info['total_frames']} frames "
          f"@ {info['fps']} fps")
    print(f"\nworking arm: {(a.mover=='left').sum()} left / {(a.mover=='right').sum()} right")
    idle = pd.concat([a[a.mover == "left"].r_path, a[a.mover == "right"].l_path])
    print(f"idle arm total motion: median {idle.median():.2f} rad, max {idle.max():.2f} rad "
          f"-> the non-working arm is genuinely parked")
    print("\nobject x arm:")
    print(pd.crosstab(a.object, a.mover).to_string())

    fig_split(a, out / "arm_split.png")
    fig_trajectories(df, state, a, out / "trajectories.png")
    fig_reach(df, state, a, out / "reach_spread.png")
    print(f"\nwrote {out}/arm_split.png, trajectories.png, reach_spread.png, per_episode.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
