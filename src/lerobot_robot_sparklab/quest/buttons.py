"""WebXR "xr-standard" gamepad mapping — shared across every robot.

These indices are the WebXR spec's standard layout, not anything about a
particular arm, so they belong here rather than in a robot's teleoperator.

A controller that reports a SHORTER button array reads as "not pressed"
forever at every index past its end — silently. That failure mode has bitten
us (a stow button that simply never fired, with nothing in the log to say
why), which is why :func:`describe_layout` exists: call it once per hand and
log the result, so "the button does nothing" is distinguishable from "the
button never reached us".
"""

from __future__ import annotations

# --- xr-standard indices ---------------------------------------------------
TRIGGER = 0      # analog 0..1 — usually gripper closure
GRIP = 1         # boolean    — usually the clutch
THUMBSTICK = 3   # boolean    — thumbstick press
A_X = 4          # boolean    — A (right hand) / X (left hand)
B_Y = 5          # boolean    — B (right hand) / Y (left hand)

# How long an xr_frame may go unrefreshed before its controller pose is
# considered stale. Above this, a pose is from before the gap and acting on
# it would produce catch-up motion.
XR_FRAME_STALE_TIMEOUT_S = 0.2


def pressed(buttons: list | None, index: int) -> bool:
    """Boolean state of ``index``, False if the controller doesn't report it.

    Out-of-range is treated as "not pressed" rather than raising, because a
    controller legitimately may not have that button — but see
    :func:`describe_layout` for making that visible instead of mysterious.
    """
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

    Names any ``required`` index the controller cannot report, so a button
    that can never fire is visible immediately rather than being mistaken for
    broken downstream logic.
    """
    n = len(buttons) if buttons else 0
    out_of_range = [i for i in required if i >= n]
    msg = f"controller reports {n} buttons"
    if out_of_range:
        msg += f" — indices {out_of_range} are OUT OF RANGE and can never fire"
    return msg
