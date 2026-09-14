"""One coordinate/preprocessing contract for labels, training, and inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = 256
MAP_SIZE = 64
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Letterbox:
    size: int
    left: int
    top: int
    width: int
    height: int

    @classmethod
    def fit(cls, width: int, height: int, size: int = INPUT_SIZE) -> "Letterbox":
        if min(width, height, size) <= 0:
            raise ValueError("Image and input dimensions must be positive")
        scale = min(size / width, size / height)
        # Match Go math.Round, including .5 ties (Python round uses ties-to-even).
        w = max(1, min(size, int(width * scale + 0.5)))
        h = max(1, min(size, int(height * scale + 0.5)))
        return cls(size, (size - w) // 2, (size - h) // 2, w, h)

    def coverage(self, map_size: int = MAP_SIZE) -> np.ndarray:
        """Fraction of each map cell occupied by image content."""
        edges = np.linspace(0, self.size, map_size + 1)
        x = np.clip(
            np.minimum(edges[1:], self.left + self.width) - np.maximum(edges[:-1], self.left),
            0,
            None,
        )
        y = np.clip(
            np.minimum(edges[1:], self.top + self.height) - np.maximum(edges[:-1], self.top),
            0,
            None,
        )
        return (np.outer(y, x) / (self.size / map_size) ** 2).astype(np.float32)


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError(f"Image exceeds 20 megapixels: {path}")
        return ImageOps.exif_transpose(source).convert("RGBA")


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        p.resolve()
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No JPEG, PNG, or WebP images found in {directory}")
    return paths


def prepare_image(
    image: Image.Image, size: int = INPUT_SIZE, *, bgr: bool = False
) -> tuple[np.ndarray, Letterbox]:
    """Aspect-fit RGBA, composite on black, return float32 NCHW (batch=1).

    RGB uses ImageNet normalization and zero padding. YuNet's BGR variant
    uses raw 0..255 values and black padding, as in Autogravity.
    Call open_image first when EXIF orientation must be applied.
    """
    box = Letterbox.fit(image.width, image.height, size)
    resized = image.convert("RGBA").resize((box.width, box.height), Image.Resampling.BILINEAR)
    rgba = np.asarray(resized, dtype=np.uint32)
    if bgr:
        pixels = ((rgba[..., :3] * rgba[..., 3:4]) // 255).astype(np.float32)[..., ::-1]
    else:
        rgb16 = (rgba[..., :3] * 257 * rgba[..., 3:4]) // 255
        pixels = (rgb16.astype(np.float32) / 65535 - MEAN) / STD
    tensor = np.zeros((1, 3, size, size), dtype=np.float32)
    tensor[0, :, box.top : box.top + box.height, box.left : box.left + box.width] = (
        pixels.transpose(2, 0, 1)
    )
    return tensor, box


def resize_map(values: np.ndarray, size: int = MAP_SIZE) -> np.ndarray:
    return np.array(
        Image.fromarray(np.asarray(values, dtype=np.float32)).resize(
            (size, size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def _sample(values: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sampling in pixel-center coordinates with edge extension."""
    xs = np.clip(xs, 0, values.shape[1] - 1)
    ys = np.clip(ys, 0, values.shape[0] - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, values.shape[1] - 1), np.minimum(y0 + 1, values.shape[0] - 1)
    wx, wy = xs - x0, (ys - y0)[:, None]
    a = values[np.ix_(y0, x0)] * (1 - wx) + values[np.ix_(y0, x1)] * wx
    b = values[np.ix_(y1, x0)] * (1 - wx) + values[np.ix_(y1, x1)] * wx
    return (a * (1 - wy) + b * wy).astype(np.float32)


def project_map(values: np.ndarray, box: Letterbox) -> tuple[np.ndarray, np.ndarray]:
    """Project an oriented-image map into the student's letterboxed map grid."""
    centers = (np.arange(MAP_SIZE) + 0.5) * box.size / MAP_SIZE
    xs = (centers - box.left) / box.width * values.shape[1] - 0.5
    ys = (centers - box.top) / box.height * values.shape[0] - 0.5
    valid = box.coverage()
    target = _sample(values, xs, ys)
    target[valid == 0] = 0
    return target, valid


def restore_map(values: np.ndarray, box: Letterbox, size: int = MAP_SIZE) -> np.ndarray:
    """Remove padding, returning a map in normalized oriented-image coordinates."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Expected a square 2D map")
    grid = values.shape[0]
    centers = (np.arange(size) + 0.5) / size
    xs = (box.left + centers * box.width) / box.size * grid - 0.5
    ys = (box.top + centers * box.height) / box.size * grid - 0.5
    valid = (box.coverage(grid) > 0).astype(np.float32)
    # Normalize interpolation at boundaries so activations wholly in padding
    # cannot pull the focal point out of the image.
    numerator = _sample(np.where(valid > 0, values, 0), xs, ys)
    return numerator / np.maximum(_sample(valid, xs, ys), 1e-8)
