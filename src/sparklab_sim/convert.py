"""Convert the vendored YAM-Ultra URDF to USD. Run once; the result is cached.

    <isaac>/python.sh -m sparklab_sim.convert

Boots a headless SimulationApp because the URDF importer is a Kit extension,
which is also why every Isaac import here sits inside ``main()``.

The source is the vendored URDF, not i2rt's installed copy: that clone predates
v1.2.4 and still has the swapped joint2/joint3 limits, which would give the
twin an elbow travelling 0.66 rad past the real stop.
"""

from __future__ import annotations

import argparse

from . import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="reconvert even if the USD already exists")
    ap.add_argument("--headless", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    urdf = paths.require(paths.YAM_ULTRA_URDF)
    if paths.YAM_ULTRA_USD.exists() and not args.force:
        print(f"already converted: {paths.YAM_ULTRA_USD}\n(use --force to redo)")
        return 0

    paths.USD_DIR.mkdir(parents=True, exist_ok=True)

    # --- Isaac boots here; nothing omni/isaacsim may be imported above ------
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})

    try:
        import omni.kit.app
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        # The importer leans on both of these; the shipped example enables
        # them explicitly rather than relying on load order.
        ext_manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
        ext_manager.set_extension_enabled_immediate("isaacsim.robot.schema", True)

        from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig

        cfg = URDFImporterConfig()
        cfg.urdf_path = str(urdf)
        cfg.usd_path = str(paths.USD_DIR)
        # Anchor the arm to the world: it is bolted to a table, and a
        # floating base would sag under gravity the moment physics ran.
        cfg.fix_base = True
        # Merging fixed joints is a rendering optimisation that erases link
        # frames — exactly what wrist-camera extrinsics will attach to.
        cfg.merge_fixed_joints = False
        # Kinematic posing only, so no drive tuning is meaningful here. Set
        # position targets so the joints are drivable when physics is off.
        cfg.joint_target_type = "position"

        print(f"converting {urdf.name} -> {paths.USD_DIR}")
        out = URDFImporter(cfg).import_urdf()
        if not out:
            print("ERROR: importer returned no output path")
            return 1
        print(f"wrote {out}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
