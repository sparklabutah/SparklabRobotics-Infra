"""Measured layout of the bimanual YAM-Ultra rig.

Everything here describes where physical things sit on the table. It is the
part of a digital twin that cannot be derived from the URDF, only measured —
so each value is marked MEASURED or ASSUMED. Treat the ASSUMED ones as
placeholders that will produce a plausible-looking but wrong scene, and
replace them as they get measured.

Frame convention (matches Isaac's default Z-up stage):

    +X  forward, the direction both arms reach       (arm rest pose puts
                                                      tool0 at +X)
    +Y  left, the axis the two arms are spread along
    +Z  up
    origin: table surface, midway between the two arm bases — i.e. directly
            under the third-person camera.
"""

from __future__ import annotations

import numpy as np

# ── MEASURED ────────────────────────────────────────────────────────────────

# Outer span: leftmost edge of one arm's base plate to the rightmost edge of
# the other's. Measured on the rig.
ARM_OUTER_SPAN_M = 0.80

# Base plate footprint, read off assets/base.stl's bounding box:
#   X 0.0794 m (deep) x  Y 0.2000 m (wide)  x  Z 0.07 m (tall)
# The arms are spread along Y, so the width that matters is the Y extent.
BASE_WIDTH_M = 0.200

# Centre-to-centre spacing follows: the outer span covers both half-widths
# plus the gap between centres.
ARM_SEPARATION_M = ARM_OUTER_SPAN_M - BASE_WIDTH_M          # 0.600 m

# Arm base origins on the table. +Y is left.
LEFT_ARM_POSITION = np.array([0.0, +ARM_SEPARATION_M / 2, 0.0])
RIGHT_ARM_POSITION = np.array([0.0, -ARM_SEPARATION_M / 2, 0.0])

# The D435F third-person camera sits exactly midway between the two arms.
# ONLY the lateral (Y) placement is measured — it is centred, hence the
# choice of origin above. How far back it stands and how high it is mounted
# are not measured; see TOP_CAMERA_STANDOFF_M / TOP_CAMERA_HEIGHT_M below.
TOP_CAMERA_Y = 0.0


# ── ASSUMED — replace with measurements ─────────────────────────────────────

# Both arms pointing straight down +X, no inward toe-in. Many bimanual rigs
# angle the arms toward each other to enlarge the shared workspace; if this
# one does, this is wrong and every rendered frame is subtly off.
LEFT_ARM_YAW_RAD = 0.0
RIGHT_ARM_YAW_RAD = 0.0

# Third-person camera standoff (how far back along -X it stands), height
# above the table, and downward pitch. These three are still guesses chosen so
# a demo render frames both arms — NOT the real rig's values. Placing the
# camera at the exact midpoint with no standoff puts both arms outside the
# frame entirely, which is how they got their current numbers.
TOP_CAMERA_STANDOFF_M = 1.10
TOP_CAMERA_HEIGHT_M = 0.85
TOP_CAMERA_PITCH_RAD = np.deg2rad(-22.0)

# The lens, unlike the placement, is MEASURED: read off D435F serial
# 243322071190 on its native 640x360 colour stream via pyrealsense2, giving
# fx = 455.570 (HFOV 70.17 deg). Converted against the 20.955 mm horizontal
# aperture the scene uses:  455.570 * 20.955 / 640.
# Was 18.0, a guess worth 3.5 degrees of horizontal field.
TOP_CAMERA_FOCAL_MM = 14.9164

# Table top dimensions, for the visual ground slab only. Does not affect
# kinematics.
TABLE_SIZE_M = (1.20, 1.60)

# Wrist cameras: mounted rigidly to each link6, so their pose follows FK once
# the mount transform is known. That transform is a hand-eye calibration that
# has not been done — the URDF contains no camera links at all
# (`grep -ci camera yam_ultra.urdf` -> 0). Left unset rather than guessed,
# because a wrong extrinsic is a first-order error that no amount of
# geometric fidelity compensates for.
WRIST_CAMERA_MOUNT = None


def summary() -> str:
    """One-block report of the layout, flagging what is still assumed."""
    return "\n".join([
        "bimanual YAM-Ultra layout",
        f"  MEASURED  outer span            {ARM_OUTER_SPAN_M:.3f} m",
        f"  DERIVED   base width (mesh)     {BASE_WIDTH_M:.3f} m",
        f"  DERIVED   arm separation (c-c)  {ARM_SEPARATION_M:.3f} m",
        f"  MEASURED  top camera centred    y={TOP_CAMERA_Y:.3f} m",
        f"  ASSUMED   arm yaw L/R           "
        f"{LEFT_ARM_YAW_RAD:.3f} / {RIGHT_ARM_YAW_RAD:.3f} rad",
        f"  ASSUMED   top camera standoff   {TOP_CAMERA_STANDOFF_M:.3f} m\n"
        f"  ASSUMED   top camera height     {TOP_CAMERA_HEIGHT_M:.3f} m",
        f"  ASSUMED   top camera pitch      {np.rad2deg(TOP_CAMERA_PITCH_RAD):.1f} deg",
        "  MISSING   wrist camera extrinsics (hand-eye calibration not done)",
    ])
