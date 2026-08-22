"""An ordinary LeRobot rollout plus a control port, and a REPL to drive it.

    live.py   the rollout loop + an HTTP control channel (POST /cmd)
    ctl.py    REPL client for that channel

WHY THIS IS NOT IN ``sparklab_sim``. It used to be, and that was wrong twice
over. It imports lerobot, which Isaac's bundled Python does not have, so it
could never run under the interpreter that package exists to serve; and it
drives real arms just as often as simulated ones -- ``rollout.sh --mode=live``
defaults to ``--robot=hw``. Its ramps go through the follower's Δq clamp
exactly like a policy action, which is why it was written to drive both.

The sim connection is only that ``--robot.type=yam_ultra_sim`` is one of the
robots it can be pointed at, the same way ``yam_ultra_bimanual`` is.
"""
