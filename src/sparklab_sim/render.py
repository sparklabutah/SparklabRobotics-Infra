"""Render the scene's cameras to numpy arrays.

Holds the Replicator render product and annotator so they are created **once**
per camera. Creating them per frame is slow enough to make interactive scene
tuning miserable, which is the whole thing this module exists to avoid.
"""

from __future__ import annotations

import numpy as np


class Renderer:
    """One render product bound to one camera prim.

        r = Renderer(app, scene.CAMERA_PRIM)
        frame = r.frame()          # (H, W, 3) uint8 RGB

    Reuse it across tweaks: change the camera or the arm pose, call
    ``frame()`` again. Do not build a new one per iteration.
    """

    def __init__(self, app, camera_prim: str, resolution=(640, 360),
                 settle_frames: int = 12):
        import omni.replicator.core as rep

        self._app = app
        self._settle = settle_frames
        self._product = rep.create.render_product(camera_prim, resolution)
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb.attach(self._product)
        # First frame after attach is usually blank/partial — RTX needs a few
        # updates to converge. Pay that cost once, here.
        self._pump(self._settle)

    def _pump(self, n: int) -> None:
        for _ in range(n):
            self._app.update()

    def frame(self, settle: int | None = None) -> np.ndarray:
        """Current camera image as (H, W, 3) uint8 RGB.

        ``settle`` controls how many app updates run before reading. Lower it
        (2-3) for fast interactive tweaking where a slightly noisy image is
        fine; raise it for a final capture.
        """
        self._pump(self._settle if settle is None else settle)
        return np.asarray(self._rgb.get_data())[..., :3].astype(np.uint8)

    def close(self) -> None:
        """Release the render product. **Call this before reopening a stage.**

        Not optional and not merely a leak. A render product is a prim under
        ``/Render/OmniverseKit/HydraTextures/``; reopening or reloading the
        stage deletes those prims while Hydra still holds pointers to them,
        and the next update dereferences freed memory:

            [Error] [rtx.hydra] Invalid USD RenderProduct Prim: .../Replicator_01
            [Error] [omni.hydra] Unable to find RP Prim from previous update pass!
            Segmentation fault

        That crash is why this method exists. It used to be absent, and the
        live viewer's teardown probed for ``close``/``destroy``/``detach`` on
        *this* object, found none of them (``destroy`` is on the underlying
        HydraTexture, not here), and silently did nothing — so every reload
        took the process down and cost a full Isaac boot to recover.

        Idempotent: safe to call twice, and safe on a partially built object.
        """
        rgb, product = getattr(self, "_rgb", None), getattr(self, "_product", None)
        # Detach before destroy: the annotator holds its own reference to the
        # product, and destroying underneath it leaves the registry with a
        # dangling entry that resurfaces on the next attach.
        if rgb is not None and product is not None:
            try:
                rgb.detach([product])
            except Exception:
                pass
        if product is not None:
            try:
                product.destroy()
            except Exception:
                pass
        self._rgb = self._product = None


def side_by_side(*images: np.ndarray, gap: int = 8) -> np.ndarray:
    """Lay images out horizontally with a separator, for real-vs-render diffs.

    Images must share a height. Mismatched heights are a bug worth surfacing
    loudly rather than silently letterboxing — a rendered frame that is not
    the same shape as the recorded one means the camera resolution is wrong.
    """
    if not images:
        raise ValueError("nothing to lay out")
    h = images[0].shape[0]
    for i, im in enumerate(images):
        if im.shape[0] != h:
            raise ValueError(
                f"image {i} has height {im.shape[0]}, expected {h} — a render "
                f"that mismatches the recorded frame usually means the camera "
                f"resolution does not match the dataset's 640x360")
    sep = np.full((h, gap, 3), 40, dtype=np.uint8)
    out = []
    for i, im in enumerate(images):
        if i:
            out.append(sep)
        out.append(im)
    return np.concatenate(out, axis=1)


def grid(images: dict[str, np.ndarray], cols: int = 2,
         label: bool = True) -> np.ndarray:
    """Tile ``{name: image}`` into one labelled image.

    Used to put every camera plus the workspace overview into a single MJPEG
    stream, so one browser tab shows the whole scene rather than needing a tab
    per camera. Cells are padded to a common size instead of resized — a
    camera whose resolution has drifted from the dataset's should look wrong,
    not be silently scaled to fit.
    """
    import cv2

    if not images:
        raise ValueError("nothing to tile")
    names = list(images)
    h = max(images[n].shape[0] for n in names)
    w = max(images[n].shape[1] for n in names)

    cells = []
    for n in names:
        im = images[n]
        cell = np.zeros((h, w, 3), dtype=np.uint8)
        cell[: im.shape[0], : im.shape[1]] = im
        if label:
            cv2.rectangle(cell, (0, 0), (w, 22), (0, 0, 0), -1)
            cv2.putText(cell, f"{n}  {im.shape[1]}x{im.shape[0]}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                        cv2.LINE_AA)
        cells.append(cell)

    cols = max(1, min(cols, len(cells)))
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        while len(row) < cols:                       # pad a ragged last row
            row.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def abs_diff(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Absolute difference image and mean absolute error (0-255).

    The scalar is what turns "looks about right" into something you can watch
    go down while tuning.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return d.astype(np.uint8), float(d.mean())
