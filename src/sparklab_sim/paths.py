"""Locate shared assets in the sibling package — by path, never by import.

``lerobot_robot_sparklab`` imports lerobot at package-import time, which does
not exist under Isaac Sim's bundled Python. So the model files are resolved
as plain paths instead. Same repo, same single copy of the URDF and meshes,
no import coupling.
"""

from __future__ import annotations

from pathlib import Path

# src/sparklab_sim/paths.py -> src/
_SRC = Path(__file__).resolve().parent.parent

YAM_ULTRA_MODEL_DIR = _SRC / "lerobot_robot_sparklab" / "robots" / "yam_ultra" / "model"
YAM_ULTRA_URDF = YAM_ULTRA_MODEL_DIR / "yam_ultra.urdf"
YAM_ULTRA_CONFIG_DIR = _SRC / "lerobot_robot_sparklab" / "robots" / "yam_ultra" / "config"

# Converted USD lands beside the source model, not in a temp dir: conversion is
# slow, the result is deterministic, and the notebook should be re-runnable
# without redoing it. Gitignored — it is a build artifact, not a source asset.
USD_DIR = YAM_ULTRA_MODEL_DIR / "usd"
# The importer creates <usd_path>/<stem>/<stem>.usda plus a payloads/ tree, so
# the file is one level deeper than the directory handed to it.
YAM_ULTRA_USD = USD_DIR / "yam_ultra" / "yam_ultra.usda"


def require(path: Path) -> Path:
    """Return *path*, or raise with a message that says what to do about it."""
    if not path.exists():
        if path == YAM_ULTRA_USD:
            raise FileNotFoundError(
                f"{path} not found — run the URDF->USD conversion first:\n"
                f"    <isaac>/python.sh -m sparklab_sim.convert")
        raise FileNotFoundError(f"missing asset: {path}")
    return path
