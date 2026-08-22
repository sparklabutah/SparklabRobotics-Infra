"""Isaac Sim digital twin of the SparkLab rigs — kinematic, not dynamic.

Poses the arms from recorded joint trajectories and renders the rig's cameras
for synthetic data and offline evaluation. Physics is never stepped: no
contact, torque or grasping. See DESIGN.md for that scope decision, and for
why this is a separate top-level package.

Runs under Isaac's bundled Python, not a conda env::

    <isaac>/python.sh -m sparklab_sim.convert

``SimulationApp`` must be constructed before any ``omni.*`` / ``isaacsim.*``
import, so no module here may import them at module scope.
"""

__version__ = "0.1.0"
