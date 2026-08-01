"""One subpackage per robot in the lab.

Each robot owns everything specific to it — kinematics/MJCF, IK, the
hardware bridge to its motor driver, its LeRobot Robot adapter, its
teleoperator, and its rig config (camera serials, CAN channels). Anything
genuinely reusable belongs one level up in relay/, quest/, core/, cameras/
or tools/ instead.
"""
