"""Isaac Sim digital twin of the SparkLab rigs — kinematic, not dynamic.

WHAT THIS IS FOR
================
Synthetic data and offline evaluation: pose the arms from recorded joint
trajectories, render the rig's cameras, vary lighting/texture/viewpoint.

It deliberately does **not** step physics. Nothing here simulates contact,
torque or grasping; the arms are posed kinematically. That is a scope
decision, not a missing feature — the policy consumes RGB plus joint state
and emits joint positions, so nothing in that loop asks the simulator what
force a gripper applies. Skipping dynamics also skips the system
identification (joint friction, damping, drive gains, contact parameters)
that none of the recorded data can supply.

WHY THIS IS A SEPARATE TOP-LEVEL PACKAGE
========================================
It runs under Isaac Sim's *bundled* Python (3.12, in the standalone install),
which has no ``lerobot``. Importing ``lerobot_robot_sparklab`` executes its
``__init__`` — which imports lerobot to register the robot plugins — and
would fail immediately. So nothing here imports that package.

Shared assets (the URDF, meshes, rig config) are still single-source: this
package reads them **by path** out of the sibling package's tree. See
``paths.py``. One repo, one copy of the model, two interpreters.

That makes the boundary a mechanical test rather than a judgement call: **if a
module imports lerobot, it belongs in the sibling package, not here.** The
rollout runner and its REPL failed that test and now live in
``lerobot_robot_sparklab.rollout``.

RUNNING
=======
Use the standalone's Python, not a conda env::

    ~/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/python.sh -m sparklab_sim.convert

For notebooks, ``./jupyter_notebook.sh`` from that directory registers an
"Isaac Sim Python 3" kernel and launches Jupyter against it.

IMPORT ORDER IS LOAD-BEARING
============================
``SimulationApp`` must be constructed **before** any ``omni.*`` or
``isaacsim.*`` module is imported — those modules are provided by the
extension system and do not exist until the app has booted. So no module
here may import them at module scope; every such import lives inside a
function, after the app is up.
"""

__version__ = "0.1.0"
