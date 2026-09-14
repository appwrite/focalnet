"""Pure NumPy crop candidate generation shared by training and inference."""

import math

import numpy as np

MAX_CANDIDATES = 128


def generate_candidates(
    image_width: int,
    image_height: int,
    aspect_ratio: float,
    *,
    positions: int = 5,
    scales: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.65),
) -> np.ndarray:
    """Generate exact-ratio source-normalized crops with multiple positions and zoom levels."""
    if min(image_width, image_height, positions) <= 0:
        raise ValueError("Image dimensions and candidate positions must be positive")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be finite and positive")
    if not scales or any(not math.isfinite(scale) or not 0 < scale <= 1 for scale in scales):
        raise ValueError("Candidate scales must be finite values in (0, 1]")
    source_ratio = image_width / image_height
    if source_ratio > aspect_ratio:
        maximum_width, maximum_height = aspect_ratio / source_ratio, 1.0
    else:
        maximum_width, maximum_height = 1.0, source_ratio / aspect_ratio
    candidates = []
    for scale in scales:
        width, height = maximum_width * scale, maximum_height * scale
        xs = np.linspace(0, 1 - width, positions if width < 1 - 1e-7 else 1)
        ys = np.linspace(0, 1 - height, positions if height < 1 - 1e-7 else 1)
        for top in ys:
            for left in xs:
                candidates.append((left, top, left + width, top + height))
    result = np.unique(np.round(np.asarray(candidates, dtype=np.float32), 7), axis=0)
    if len(result) > MAX_CANDIDATES:
        raise ValueError(f"Generated {len(result)} crops; maximum is {MAX_CANDIDATES}")
    return result


def letterbox_content(box) -> np.ndarray:
    """Return normalized content bounds consumed by the crop-ranking model."""
    return (
        np.asarray(
            [box.left, box.top, box.left + box.width, box.top + box.height], dtype=np.float32
        )
        / box.size
    )
