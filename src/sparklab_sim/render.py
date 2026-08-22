"""Render the scene's cameras to numpy arrays.

Holds the Replicator render product and annotator so they are created once per
camera rather than per frame, which is slow enough to make tuning miserable.
"""

from __future__ import annotations

import numpy as np


class Renderer:
    """One render product bound to one camera prim::

        r = Renderer(app, scene.CAMERA_PRIM)
        frame = r.frame()          # (H, W, 3) uint8 RGB

    Reuse it across tweaks rather than building one per iteration.
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

        settle: app updates to run before reading. 2-3 is fine for interactive
            tweaking; raise it for a final capture.
        """
        self._pump(self._settle if settle is None else settle)
        return np.asarray(self._rgb.get_data())[..., :3].astype(np.uint8)

    def close(self) -> None:
        """Release the render product. **Call this before reopening a stage.**

        A render product is a prim under ``/Render/OmniverseKit/HydraTextures/``;
        reloading the stage deletes it while Hydra still holds a pointer, and
        the next update segfaults on freed memory.

        Idempotent: safe to call twice, and on a partially built object.
        """
        rgb, product = getattr(self, "_rgb", None), getattr(self, "_product", None)
        # Detach first: the annotator holds its own reference, and destroying
        # underneath it leaves a dangling registry entry.
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

    Images must share a height: a mismatch means the camera resolution is
    wrong, which is worth raising rather than letterboxing away.
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

    Puts every camera into a single MJPEG stream. Cells are padded rather than
    resized, so a camera whose resolution has drifted looks wrong.
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
    """Absolute difference image and mean absolute error (0-255)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return d.astype(np.uint8), float(d.mean())
