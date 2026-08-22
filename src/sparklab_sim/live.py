"""Edit a scene ``.usda`` in your editor, orbit it in a browser, over SSH.

    ./scripts/isaac_python.sh -m sparklab_sim.live

Boots Isaac once and serves an interactive viewport on one TCP port. Drag to
orbit, wheel to zoom, shift-drag to pan. Save the ``.usda`` in any editor and
the view updates in place — ``.usda`` is ASCII, so moving a prim is editing
its ``xformOp:translate``. The camera dropdown switches between free look and
the cameras authored in the scene, so you can jump from "what am I moving" to
"what does the rig actually see".

HOW THE FAST PATH WORKS, AND WHY IT IS SHAPED LIKE THIS

The scene file is loaded as a **sublayer** of an anonymous wrapper stage
rather than opened directly:

    wrapper stage (anonymous root)
      |- session layer : free camera, overview camera   (never saved)
      `- sublayer      : scene.usda                     (the file you edit)

Reloading then means ``Sdf.Layer.Reload()`` on that one sublayer, which
recomposes the scene while leaving everything else on the stage alone —
including the Replicator render products under ``/Render``. Measured on this
workstation: **~0.04 s** per edit, against ~0.38 s to tear down and reopen the
stage, against a full Isaac boot (8-30 s) for the version of this file that
called ``open_stage`` with live render products still attached and segfaulted
every time. Render products are prims too; reopening or reloading the *root*
layer deletes them under Hydra's feet:

    [Error] [rtx.hydra] Invalid USD RenderProduct Prim: .../Replicator_01
    [Error] [omni.hydra] Unable to find RP Prim from previous update pass!
    Segmentation fault

Hence: authoring cameras into the session layer, editing content in a
sublayer, and ``Renderer.close()`` before any reopen. ``--reopen`` falls back
to the slower full reopen for the cases a sublayer reload cannot express.

WHAT ``--play`` ACTUALLY DOES
It makes the viewer re-render continuously; it does not start physics. The
timeline is *always* playing, because ``scene.start()`` has to play it before
Isaac will accept joint writes at all.

That used to be harmless -- the scene had gravity off and nothing was
simulated, so a prop's authored transform WAS its rendered one. It is not
harmless now: gravity is on (9.81) and the props are dynamic rigid bodies, so
PhysX owns their poses and writes the simulated transform back every step.
A layer reload changes the authored value and PhysX overwrites it, which looks
like "I edited the position and the viewer ignored me".

``reload_layers`` therefore bounces the timeline (``scene.start(restart=True)``)
after reloading, because stopping is what resets simulated prims to their
authored state. The arms reset with them -- unavoidable, and fine here, since
this viewer exists for placing things rather than holding a pose.

So leave ``--play`` off while you are placing things — with a static scene,
re-rendering an identical frame only burns the GPU — and turn it on when
something in the stage really is moving and you want to watch it.
"""

from __future__ import annotations

import argparse
import hashlib
import signal
import sys
import time
from pathlib import Path

# How many consecutive polls a file's fingerprint must hold steady before we
# act on it. Editors that write non-atomically (or write twice) would otherwise
# reload against a half-written file. Two polls is enough in practice and costs
# one poll of latency.
STABLE_POLLS = 2

FREE = "__free__"          # sentinel the browser sends for the free-look camera


class LayerWatcher:
    """Fingerprints every on-disk layer the stage composes from.

    Watching only the top-level scene file misses the case that actually bites:
    ``scene.usda`` references ``yam_ultra.usda``, so editing the arm asset --
    a joint limit, a mesh path, the wrist camera mount -- changes what renders
    while the scene file's mtime never moves. ``Stage.GetUsedLayers()`` gives
    the real dependency set after composition, so the watch list is rebuilt on
    every reload rather than guessed up front.

    Fingerprint is (mtime_ns, size) rather than mtime alone: some filesystems
    quantise mtime to a full second, and two saves inside that window would be
    indistinguishable. Content hashing is available via ``--hash`` for network
    filesystems where even mtime_ns is unreliable -- VAST and NFS both lie
    about timestamps under some configurations.
    """

    def __init__(self, use_hash: bool = False):
        self.use_hash = use_hash
        self.paths: list[Path] = []
        self._last: dict[Path, object] = {}
        self._stable_count = 0
        self._pending: dict[Path, object] | None = None

    def track(self, paths: list[Path]) -> None:
        self.paths = paths
        self._last = self._snapshot()
        self._pending = None
        self._stable_count = 0

    def _fingerprint(self, p: Path):
        try:
            st = p.stat()
        except OSError:
            return None
        if not self.use_hash:
            return (st.st_mtime_ns, st.st_size)
        try:
            return hashlib.blake2b(p.read_bytes(), digest_size=16).hexdigest()
        except OSError:
            return None

    def _snapshot(self) -> dict[Path, object]:
        return {p: self._fingerprint(p) for p in self.paths}

    def changed(self) -> list[Path]:
        """Layers whose contents have settled since the last change.

        Returns the changed paths rather than a bare bool so the caller can
        reload precisely those layers instead of everything -- the difference
        between recomposing one small scene file and re-reading the arm asset
        and its whole payload tree.
        """
        now = self._snapshot()
        if now == self._last:
            self._pending = None
            self._stable_count = 0
            return []

        if now == self._pending:
            self._stable_count += 1
            if self._stable_count >= STABLE_POLLS:
                hit = [p for p, f in now.items() if self._last.get(p) != f]
                self._last = now
                self._pending = None
                self._stable_count = 0
                return hit
        else:
            self._pending = now
            self._stable_count = 1
        return []


def _disk_layers(stage) -> list[Path]:
    """Resolved on-disk paths for every layer in the stage's composition."""
    out = []
    for layer in stage.GetUsedLayers():
        ident = layer.realPath or layer.identifier
        if not ident or ident.startswith(("omniverse://", "anon:")):
            continue
        p = Path(ident)
        if p.exists():
            out.append(p)
    return sorted(set(out))


def _dispose(renderers: dict) -> None:
    """Release every render product. See ``render.Renderer.close``."""
    for r in renderers.values():
        try:
            r.close()
        except Exception as e:
            print(f"  dispose: {type(e).__name__}: {e}")


def main() -> int:
    # This is meant to be left running under tmux with its output redirected,
    # and Python block-buffers a redirected stdout -- which silently swallowed
    # every reload message during testing. Line buffering costs nothing here
    # and makes the log usable while the process is still alive.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None,
                    help="scene .usda (default: robots/yam_ultra/sim/scene.usda)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from another machine directly")
    ap.add_argument("--poll", type=float, default=0.1,
                    help="seconds between change checks. Save-to-pixels is "
                         "STABLE_POLLS * poll + reload; with reload down at "
                         "~0.12 s the debounce is now the larger half, which "
                         "is why this is 0.1 and not 0.5")
    ap.add_argument("--settle", type=int, default=6,
                    help="app updates per frame when idle (quality)")
    ap.add_argument("--drag-settle", type=int, default=1,
                    help="app updates per frame while orbiting (latency)")
    ap.add_argument("--res", default="960x540",
                    help="free-look render resolution, WxH")
    ap.add_argument("--cam-res", default="640x360",
                    help="resolution for cameras authored in the scene. "
                         "Defaults to the dataset's 640x360 so a rendered "
                         "frame is directly comparable to a recorded one; "
                         "rendering a rig camera at the free-look resolution "
                         "would change its aspect and make that comparison a lie")
    ap.add_argument("--play", action="store_true",
                    help="re-render continuously (default: render only on "
                         "change). Does NOT start physics — the timeline is "
                         "always playing and gravity is off; see the module "
                         "docstring")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="render rate with --play")
    ap.add_argument("--hash", action="store_true",
                    help="content-hash instead of mtime; for network filesystems")
    ap.add_argument("--reopen", action="store_true",
                    help="full open_stage per change instead of sublayer "
                         "reload; slower but recomposes from scratch")
    ap.add_argument("--once", action="store_true",
                    help="render one frame, write it, exit (for CI / snapshots)")
    ap.add_argument("--out", default="scene_preview.png",
                    help="output path for --once")
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild the scene file from rig.py, DISCARDING edits")
    args = ap.parse_args()

    try:
        rw, rh = (int(v) for v in args.res.lower().split("x"))
    except ValueError:
        ap.error(f"--res wants WxH, got {args.res!r}")
    try:
        cw, ch = (int(v) for v in args.cam_res.lower().split("x"))
    except ValueError:
        ap.error(f"--cam-res wants WxH, got {args.cam_res!r}")

    # --- Isaac boots here; no isaacsim/omni import may precede this ---------
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Sdf, Usd

    from . import paths, render, scene, viewport

    scene_path = Path(args.scene) if args.scene else (
        paths.YAM_ULTRA_MODEL_DIR.parent / "sim" / "scene.usda")

    if args.regenerate or not scene_path.exists():
        why = "regenerating" if scene_path.exists() else "no scene file yet"
        print(f"{why} -> building {scene_path} from rig.py")
        scene.export_usd(scene_path)

    watcher = LayerWatcher(use_hash=args.hash)
    renderers: dict[str, render.Renderer] = {}
    state = {"cam": None, "active": FREE}

    def authored(st) -> list[str]:
        """Cameras that came from the file, for the browser's dropdown.

        The free-look camera is on the stage but is ours, not the scene's; it
        already has its own entry, and listing it twice invites you to select
        the copy that ignores your mouse.
        """
        return list(authored_paths(st))

    def authored_paths(st) -> dict:
        return {n: p for n, p in scene.cameras(st).items()
                if p != scene.FREE_PRIM}

    # ---- stage construction ------------------------------------------------
    def open_wrapper() -> None:
        """Anonymous root stage with the scene file as its only sublayer.

        The indirection is the whole point: the scene file becomes something
        that can be reloaded independently of the stage carrying the render
        products, which is what makes an edit cost 0.04 s instead of a crash.
        """
        stage_utils.create_new_stage()
        st = stage_utils.get_current_stage()
        st.GetRootLayer().subLayerPaths = [str(scene_path)]
        app.update()

    def author_cameras() -> None:
        """Add the free-look camera to the SESSION layer.

        Session-layer edits are never saved and are not part of any sublayer,
        so this camera cannot leak into the file being edited and survives a
        sublayer reload untouched.
        """
        st = stage_utils.get_current_stage()
        with Usd.EditContext(st, st.GetSessionLayer()):
            # Re-apply the browser's last pose so a reload does not throw the
            # user back to the default view mid-edit.
            scene.set_orbit_camera(st, scene.FREE_PRIM, **(state["cam"] or {}))

    def build_renderer(name: str) -> None:
        """(Re)build the single render product for the active camera.

        One camera, not all of them: the previous version rendered every
        ``UsdGeom.Camera`` on the stage into a grid, which on a fresh Kit stage
        means four stock viewport cameras nobody asked for plus the one you
        care about -- five render products and five frames per update. Switch
        cameras from the browser instead.
        """
        nonlocal renderers
        _dispose(renderers)
        renderers = {}
        st = stage_utils.get_current_stage()
        path = (scene.FREE_PRIM if name == FREE
                else scene.cameras(st).get(name, scene.FREE_PRIM))
        if not st.GetPrimAtPath(path).IsValid():
            name, path = FREE, scene.FREE_PRIM
        res = (rw, rh) if name == FREE else (cw, ch)
        renderers = {name: render.Renderer(app, path, resolution=res,
                                           settle_frames=args.settle)}

    def load(full: bool = True) -> bool:
        """Open (or reopen) the wrapper stage and rebuild the renderer.

        Guarded end to end, not just around the open: a scene that parses but
        fails during start() -- bad physics schema, missing mesh, articulation
        with no root -- would otherwise take the process down mid-edit and cost
        an Isaac boot to get back.
        """
        try:
            if full:
                _dispose(renderers)
                open_wrapper()
            scene.start(app)
            author_cameras()
            build_renderer(state["active"])
            st = stage_utils.get_current_stage()
            watcher.track(_disk_layers(st))
            found = authored(st)
            print(f"  cameras: {', '.join(found) or '(none in file)'}"
                  f"   watching {len(watcher.paths)} layer(s)")
            return True
        except Exception as e:
            print(f"  load failed: {type(e).__name__}: {e}")
            return False

    def reload_layers(changed: list[Path]) -> bool:
        """Fast path: reload just the layers that changed on disk.

        Falls back to a full reopen when a changed file is not actually a layer
        of this composition -- a newly added reference, say, which the old
        composition never saw and so cannot reload.
        """
        if args.reopen:
            return load(full=True)
        try:
            hit = 0
            for p in changed:
                layer = Sdf.Layer.Find(str(p))
                if layer is not None:
                    layer.Reload(force=True)
                    hit += 1
            if hit != len(changed):
                return load(full=True)      # something new appeared
            app.update()
            # Bounce the timeline so PhysX rebuilds every rigid body from the
            # pose the file now says. Without this the layer reload updates the
            # authored value and the simulated one keeps being written over it,
            # so moving a prop in the .usda does nothing on screen -- the exact
            # symptom that shows up as "I changed x and it did not move".
            scene.start(app, restart=True)
            author_cameras()                # re-assert in case the file moved it
            st = stage_utils.get_current_stage()
            watcher.track(_disk_layers(st))
            return True
        except Exception as e:
            print(f"  sublayer reload failed ({type(e).__name__}: {e}); reopening")
            return load(full=True)

    def frame(settle: int):
        for r in renderers.values():
            try:
                return r.frame(settle=settle)
            except Exception as e:
                print(f"  render failed: {type(e).__name__}: {e}")
        return None

    # --- one-shot mode: no server, no loop ---------------------------------
    if args.once:
        try:
            if not load():
                return 1
            img = frame(args.settle)
            if img is None:
                print("nothing rendered")
                return 1
            try:
                import imageio.v3 as iio
                iio.imwrite(args.out, img)
            except ImportError:
                from PIL import Image
                Image.fromarray(img).save(args.out)
            print(f"wrote {args.out}")
            return 0
        finally:
            _dispose(renderers)
            app.close()

    vp = viewport.Viewport(port=args.port, host=args.host,
                           home=scene.HOME_VIEW)
    url = vp.start()
    print(f"\n  scene : {scene_path}")
    print(f"  view  : {url}")
    print(f"  mode  : {'continuous render' if args.play else 'static (render on change)'}"
          f"   reload: {'full reopen' if args.reopen else 'sublayer'}")
    print("  drag to orbit, wheel to zoom, shift-drag to pan.")
    print("  edit the .usda and save — the view follows. Ctrl-C to stop.\n")

    # SIGTERM as well as Ctrl-C: this gets run under tmux and killed by
    # scripts, and skipping the finally block leaks the port and the GPU
    # context, which then blocks the next run.
    stopping = False

    def _stop(signum, frame_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)

    try:
        if not load():
            print("initial load failed; fix the scene and save to retry")
        else:
            img = frame(args.settle)
            if img is not None:
                vp.publish(img)

        st = stage_utils.get_current_stage()
        vp.set_info(authored(st), "live")

        next_render = time.monotonic()
        next_poll = 0.0
        last_move = 0.0

        while not stopping:
            now = time.monotonic()
            drew = False

            # --- file changes ------------------------------------------------
            if now >= next_poll:
                next_poll = now + args.poll
                changed = watcher.changed()
                if vp.take_reload():
                    changed = changed or [scene_path]
                if changed or not renderers:
                    names = ", ".join(p.name for p in changed) or "(forced)"
                    print(f"[{time.strftime('%H:%M:%S')}] changed: {names}")
                    t = time.time()
                    ok = reload_layers(changed) if renderers else load(full=True)
                    if ok:
                        img = frame(args.settle)
                        if img is not None:
                            vp.publish(img)
                            drew = True
                        st = stage_utils.get_current_stage()
                        vp.set_info(authored(st),
                                    f"reloaded in {time.time()-t:.2f}s")

            # --- camera switch from the browser ------------------------------
            sel = vp.take_selection()
            if sel is not None and sel != state["active"]:
                state["active"] = sel
                build_renderer(sel)
                img = frame(args.settle)
                if img is not None:
                    vp.publish(img)
                    drew = True

            # --- orbit input --------------------------------------------------
            cam = vp.take_camera()
            if cam is not None:
                state["cam"] = cam
                if state["active"] == FREE:
                    st = stage_utils.get_current_stage()
                    with Usd.EditContext(st, st.GetSessionLayer()):
                        scene.set_orbit_camera(st, scene.FREE_PRIM, **cam)
                    img = frame(args.drag_settle)
                    if img is not None:
                        vp.publish(img)
                        drew = True
                    last_move = now

            # --- quality refinement once the mouse stops ----------------------
            # Orbiting renders at drag_settle for latency, which is visibly
            # noisier under RTX temporal sampling. When the user stops, spend
            # the extra updates once to leave a clean image on screen.
            if (not args.play and last_move and now - last_move > 0.35
                    and state["active"] == FREE):
                last_move = 0.0
                img = frame(args.settle)
                if img is not None:
                    vp.publish(img)
                    drew = True

            # --- physics ------------------------------------------------------
            if args.play:
                if now >= next_render:
                    next_render = now + 1.0 / max(args.fps, 0.1)
                    img = frame(args.drag_settle)
                    if img is not None:
                        vp.publish(img)
                        drew = True
            elif not drew:
                # With physics off the scene cannot change on its own, so
                # re-rendering an identical frame just burns the GPU. The MJPEG
                # stream holds the last published frame, so idling is free.
                # Short sleep, not `poll`: this is also the orbit input latency.
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        _dispose(renderers)
        vp.stop()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
