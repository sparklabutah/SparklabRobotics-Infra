"""An ordinary LeRobot rollout plus a control port, and a REPL to drive it.

    live.py   the rollout loop + an HTTP control channel (POST /cmd)
    ctl.py    REPL client for that channel

Drives hardware and the Isaac twin alike — only ``--robot.type`` differs.
"""
