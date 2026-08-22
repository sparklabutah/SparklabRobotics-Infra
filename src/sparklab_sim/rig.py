"""Measured layout of the bimanual YAM-Ultra rig — what the URDF cannot supply.

Each value is marked MEASURED or ASSUMED; ASSUMED ones render a plausible but
wrong scene until replaced.

Frame (Isaac's default Z-up stage): +X forward, the direction both arms reach;
+Y left, the axis they are spread along; +Z up. Origin at the table surface,
midway between the arm bases and directly under the third-person camera.
"""

from __future__ import annotations

import numpy as np

# ── MEASURED ────────────────────────────────────────────────────────────────

# Outer edge of one base plate to the outer edge of the other.
ARM_OUTER_SPAN_M = 0.80

# Y extent of the base plate's bounding box in assets/base.stl.
BASE_WIDTH_M = 0.200

# Centre-to-centre: the outer span covers both half-widths plus the gap.
ARM_SEPARATION_M = ARM_OUTER_SPAN_M - BASE_WIDTH_M          # 0.600 m

# Arm base origins on the table. +Y is left.
LEFT_ARM_POSITION = np.array([0.0, +ARM_SEPARATION_M / 2, 0.0])
RIGHT_ARM_POSITION = np.array([0.0, -ARM_SEPARATION_M / 2, 0.0])

# The D435F is centred between the arms. Only this lateral placement is
# measured; standoff and height below are not.
TOP_CAMERA_Y = 0.0


# ── ASSUMED — replace with measurements ─────────────────────────────────────

# Assumes no inward toe-in. If the rig has any, every rendered frame is off.
LEFT_ARM_YAW_RAD = 0.0
RIGHT_ARM_YAW_RAD = 0.0

# Standoff along -X, height, and downward pitch. Guesses chosen so a demo
# render frames both arms — not the real rig's values.
TOP_CAMERA_STANDOFF_M = 1.10
TOP_CAMERA_HEIGHT_M = 0.85
TOP_CAMERA_PITCH_RAD = np.deg2rad(-22.0)

# MEASURED, unlike the placement: fx = 455.570 on the D435F's native 640x360
# colour stream, converted against the scene's 20.955 mm aperture.
TOP_CAMERA_FOCAL_MM = 14.9164

# Visual ground slab only; does not affect kinematics.
TABLE_SIZE_M = (1.20, 1.60)

# Rigid to each link6, so the pose follows FK once the mount transform is
# known. That hand-eye calibration has not been done; unset beats guessed.
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
