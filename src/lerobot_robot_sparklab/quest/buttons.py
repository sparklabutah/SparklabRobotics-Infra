"""WebXR "xr-standard" gamepad mapping — shared across every robot.

A controller reporting a shorter button array reads as "not pressed" forever
past its end, silently. Log :func:`describe_layout` once per hand so that is
distinguishable from broken downstream logic.
"""

from __future__ import annotations

# --- xr-standard indices ---------------------------------------------------
TRIGGER = 0      # analog 0..1 — usually gripper closure
GRIP = 1         # boolean    — usually the clutch
THUMBSTICK = 3   # boolean    — thumbstick press
A_X = 4          # boolean    — A (right hand) / X (left hand)
B_Y = 5          # boolean    — B (right hand) / Y (left hand)

# Past this, a controller pose predates the gap and acting on it would
# produce catch-up motion.
XR_FRAME_STALE_TIMEOUT_S = 0.2


def pressed(buttons: list | None, index: int) -> bool:
    """Boolean state of ``index``, False if the controller doesn't report it."""
    if not buttons or index >= len(buttons):
        return False
    return bool(buttons[index].get("p", False))


def value(buttons: list | None, index: int) -> float:
    """Analog value of ``index`` in 0..1, 0.0 if not reported."""
    if not buttons or index >= len(buttons):
        return 0.0
    return float(buttons[index].get("v", 0.0))


def describe_layout(buttons: list | None, *required: int) -> str:
    """One-line summary for logging once per hand at startup.

    required: indices to flag if the controller cannot report them.
    """
    n = len(buttons) if buttons else 0
    out_of_range = [i for i in required if i >= n]
    msg = f"controller reports {n} buttons"
    if out_of_range:
        msg += f" — indices {out_of_range} are OUT OF RANGE and can never fire"
    return msg
