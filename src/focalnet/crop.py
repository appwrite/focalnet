"""Aspect-ratio crop selection using all the importance mass."""

import math
from dataclasses import asdict, dataclass

import numpy as np


def clean_map(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("Expected a nonempty 2D heatmap")
    return np.where(np.isfinite(values) & (values > 0), values, 0)


def focal_point(heatmap: np.ndarray) -> dict[str, float]:
    values = clean_map(heatmap)
    total = values.sum()
    if total == 0:
        return {"x": 0.5, "y": 0.5}
    h, w = values.shape
    return {
        "x": float(values.sum(axis=0) @ ((np.arange(w) + 0.5) / w) / total),
        "y": float(values.sum(axis=1) @ ((np.arange(h) + 0.5) / h) / total),
    }


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int
    retained_importance: float

    def to_dict(self) -> dict:
        return asdict(self)


def _best_offset(mass: np.ndarray, extent: int, window: int) -> tuple[int, float]:
    """Exact search over integer offsets for a piecewise-uniform map density.

    A maximal-area crop spans one whole image axis, reducing the 2D integral
    to a 1D sliding window. Its score changes slope only at heatmap cell edges
    or an edge minus the window width. Evaluate neighboring integers at each
    breakpoint, plus the centered position, to avoid an image-sized scan.
    """
    edges = np.linspace(0, extent, len(mass) + 1)
    cumulative = np.concatenate(([0.0], np.cumsum(mass)))
    breaks = np.concatenate((edges, edges - window, [(extent - window) / 2]))
    candidates = np.unique(
        np.clip(np.concatenate((np.floor(breaks), np.ceil(breaks))), 0, extent - window).astype(int)
    )
    scores = np.interp(candidates + window, edges, cumulative) - np.interp(
        candidates, edges, cumulative
    )
    best = scores.max()
    ties = np.flatnonzero(np.isclose(scores, best, rtol=1e-10, atol=0))
    center = (extent - window) / 2
    index = ties[np.argmin(np.abs(candidates[ties] - center))]
    return int(candidates[index]), float(scores[index])


def solve_crop(
    heatmap: np.ndarray, image_width: int, image_height: int, aspect_ratio: float
) -> Crop:
    """Largest inscribed crop at the requested ratio, rounded to integer pixels.

    Chooses the position retaining the most importance. Equal scores prefer
    the image center. A zero map produces a centered crop and zero retention.
    This optimizes retention, not aesthetics or face-box containment.
    """
    if min(image_width, image_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be finite and positive")
    values = clean_map(heatmap)
    width, height = image_width, image_height
    if width / height > aspect_ratio:
        width = max(1, min(width, int(height * aspect_ratio + 0.5)))
    else:
        height = max(1, min(height, int(width / aspect_ratio + 0.5)))
    if width < image_width:
        left, retained = _best_offset(values.sum(axis=0), image_width, width)
        top = 0
    else:
        top, retained = _best_offset(values.sum(axis=1), image_height, height)
        left = 0
    total = float(values.sum())
    return Crop(left, top, width, height, float(np.clip(retained / total, 0, 1)) if total else 0)


def score_crop(heatmap: np.ndarray, crop: Crop, image_width: int, image_height: int) -> float:
    """Fractional-cell integration of a crop against an independent reference map."""
    values = clean_map(heatmap)
    total = values.sum()
    if total == 0:
        return 0.0
    rows, columns = values.shape
    xs = np.linspace(0, image_width, columns + 1)
    ys = np.linspace(0, image_height, rows + 1)
    wx = np.maximum(
        0, np.minimum(xs[1:], crop.left + crop.width) - np.maximum(xs[:-1], crop.left)
    ) / (image_width / columns)
    wy = np.maximum(
        0, np.minimum(ys[1:], crop.top + crop.height) - np.maximum(ys[:-1], crop.top)
    ) / (image_height / rows)
    return float(np.clip(wy @ values @ wx / total, 0, 1))
