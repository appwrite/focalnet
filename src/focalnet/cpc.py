"""Comparative Photo Composition dataset parsing and training splits."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from focalnet.candidates import letterbox_content
from focalnet.imaging import open_image, prepare_image


def read_cpc_annotation(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read CPC's six-worker ratings and left/top/right/bottom crop coordinates."""
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, str):
            payload = json.loads(payload)
        raw_boxes = np.asarray(payload["bboxes"], dtype=np.float32)
        ratings = np.asarray(payload["scores"], dtype=np.float32)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CPC annotation: {path}") from exc
    if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 4 or len(raw_boxes) < 2:
        raise ValueError(f"{path}: expected at least two crop boxes")
    if ratings.ndim == 1 and len(ratings) == len(raw_boxes):
        scores = ratings
    elif ratings.ndim == 2 and ratings.shape[1] == len(raw_boxes):
        scores = ratings.mean(0)
    elif ratings.ndim == 2 and ratings.shape[0] == len(raw_boxes):
        scores = ratings.mean(1)
    else:
        raise ValueError(f"{path}: crop ratings do not match {len(raw_boxes)} boxes")
    if not np.isfinite(raw_boxes).all() or not np.isfinite(scores).all():
        raise ValueError(f"{path}: values must be finite")
    left, top, right, bottom = raw_boxes.T
    if not np.all((0 <= left) & (left < right) & (right <= width)) or not np.all(
        (0 <= top) & (top < bottom) & (bottom <= height)
    ):
        raise ValueError(f"{path}: crop is outside the {width}x{height} image")
    boxes = np.stack((left / width, top / height, right / width, bottom / height), axis=1)
    return boxes.astype(np.float32), scores.astype(np.float32)


def _layout(root: Path) -> tuple[Path, Path]:
    choices = (
        (root / "images" / "all_images", root / "CollectedAnnotationsRaw" / "all_labels"),
        (root / "images", root / "CollectedAnnotationsRaw"),
    )
    for images, annotations in choices:
        if images.is_dir() and annotations.is_dir():
            return images, annotations
    raise FileNotFoundError(f"Could not find CPC images and annotations under {root}")


def cpc_paths(
    root: Path, split: str, *, validation_fraction: float = 0.1, seed: int = 42
) -> list[tuple[Path, Path]]:
    if split not in {"train", "val", "all"}:
        raise ValueError("CPC split must be train, val, or all")
    if not 0 < validation_fraction < 1:
        raise ValueError("CPC validation fraction must be in (0, 1)")
    images, annotations = _layout(root)
    paths = []
    for image in sorted(
        path
        for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ):
        annotation = annotations / f"{image.name}.txt"
        if annotation.is_file():
            paths.append((image, annotation))
    if not paths:
        raise ValueError(f"No paired CPC images and annotations found under {root}")
    order = sorted(
        range(len(paths)),
        key=lambda index: hashlib.sha256(f"{seed}:{paths[index][0].name}".encode()).digest(),
    )
    validation_count = max(1, round(len(paths) * validation_fraction))
    validation = set(order[:validation_count])
    if split == "all":
        return paths
    selected = validation if split == "val" else set(order[validation_count:])
    return [path for index, path in enumerate(paths) if index in selected]


class CPCDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        augment: bool = False,
        validation_fraction: float = 0.1,
        seed: int = 42,
        excluded_hashes: set[str] | None = None,
    ):
        self.root = root
        self.split = split
        self.augment = augment
        self.records = []
        for image_path, annotation_path in cpc_paths(
            root, split, validation_fraction=validation_fraction, seed=seed
        ):
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest() if excluded_hashes else ""
            if excluded_hashes and digest in excluded_hashes:
                continue
            with Image.open(image_path) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f"CPC coordinates require unrotated pixels: {image_path}")
                boxes, scores = read_cpc_annotation(annotation_path, image.width, image.height)
            self.records.append((image_path, boxes, scores, digest))
        if not self.records:
            raise ValueError(f"No CPC records remain in the {split} split")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        path, boxes, scores, _ = self.records[index]
        image = open_image(path)
        boxes = boxes.copy()
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        tensor, content = prepare_image(image)
        return (
            torch.from_numpy(tensor[0]),
            torch.from_numpy(boxes),
            torch.from_numpy(scores.copy()),
            torch.from_numpy(letterbox_content(content)),
            path.name,
        )


def cpc_fingerprint(root: Path, *, validation_fraction: float = 0.1, seed: int = 42) -> str:
    digest = hashlib.sha256(f"cpc-v1:{validation_fraction}:{seed}".encode())
    for image, annotation in cpc_paths(root, "all"):
        digest.update(image.name.encode())
        digest.update(annotation.read_bytes())
    return digest.hexdigest()


def train_cpc_ranker(
    dataset_root: Path,
    base_checkpoint: Path,
    output: Path,
    config,
    rank_config=None,
    resume: Path | None = None,
    *,
    validation_fraction: float = 0.1,
) -> dict:
    from focalnet.human_training import train_crop_ranker

    return train_crop_ranker(
        CPCDataset(
            dataset_root,
            "train",
            augment=True,
            validation_fraction=validation_fraction,
            seed=config.seed,
        ),
        CPCDataset(
            dataset_root,
            "val",
            validation_fraction=validation_fraction,
            seed=config.seed,
        ),
        output,
        config,
        dataset_fingerprint=cpc_fingerprint(
            dataset_root, validation_fraction=validation_fraction, seed=config.seed
        ),
        dataset_name="Comparative Photo Composition",
        base_checkpoint=base_checkpoint,
        rank_config=rank_config,
        resume=resume,
    )
