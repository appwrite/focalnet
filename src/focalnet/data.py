"""Streaming label generation and explicit, group-disjoint manifests."""

import hashlib
import json
import os
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np

from focalnet.imaging import MAP_SIZE, image_paths, open_image
from focalnet.teacher import Teacher, TeacherConfig


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_manifest(path: str | Path, *, validate_files: bool = True) -> list[dict]:
    path = Path(path).resolve()
    base = path.parent
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            for field in ("image", "importance"):
                record[field] = os.path.abspath(base / record[field])
                if validate_files and not Path(record[field]).is_file():
                    raise ValueError(f"Missing {field}: {record[field]}")
            if not isinstance(record.get("group"), str) or not record["group"].strip():
                raise ValueError("Each record needs a group (same for related/duplicate images)")
            record["weight"] = float(record.get("weight", 1.0))
            if not np.isfinite(record["weight"]) or record["weight"] <= 0:
                raise ValueError("Sample weight must be finite and positive")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        records.append(record)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()
    with path.open("x") as handle:
        for source in records:
            record = dict(source)
            for field in ("image", "importance"):
                record[field] = os.path.relpath(record[field], base)
            handle.write(json.dumps(record) + "\n")


def split_manifest(
    source: Path,
    output: Path,
    validation_fraction: float,
    seed: int,
    *,
    validate_files: bool = True,
) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be in (0, 1)")
    records = read_manifest(source, validate_files=validate_files)
    groups = sorted({record["group"] for record in records})
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required for a split")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, int(len(groups) * validation_fraction + 0.5)))
    validation = set(groups[:count])
    train = [r for r in records if r["group"] not in validation]
    val = [r for r in records if r["group"] in validation]
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(output / "train.jsonl", train)
    write_manifest(output / "val.jsonl", val)
    return {"train": len(train), "validation": len(val), "seed": seed}


def label_images(
    images: Path,
    output: Path,
    saliency_model: Path,
    face_model: Path,
    config: TeacherConfig,
    *,
    threads: int = 1,
    workers: int = 1,
    provider: str = "cpu",
    resume: bool = False,
    source_paths: list[Path] | None = None,
) -> dict:
    if workers <= 0:
        raise ValueError("Label workers must be positive")
    if source_paths is None:
        paths = image_paths(images)
    else:
        root = Path(os.path.abspath(images))
        paths = [Path(os.path.abspath(path)) for path in source_paths]
        if not paths:
            raise ValueError("No source image paths were provided")
        if len(paths) != len(set(paths)):
            raise ValueError("Source image paths must be unique")
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError(f"Source image path escapes image directory: {images}")
    provenance = {
        "format_version": 1,
        "target": "face-mass-mixture-v1",
        "layout": "oriented-image",
        "map_size": MAP_SIZE,
        "preprocessing": "pillow-bilinear-letterbox-imagenet-rgb-raw-bgr-v1",
        "teacher_config": asdict(config),
        "execution_provider": provider,
        "saliency_sha256": sha256_file(saliency_model),
        "face_sha256": sha256_file(face_model),
    }
    manifest = output / "labels.jsonl"
    if resume:
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("Resume requires identical teacher models and label settings")
        existing = (
            read_manifest(manifest, validate_files=False)
            if manifest.exists() and manifest.stat().st_size
            else []
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        existing = []
    known = {record.get("image_id", Path(record["importance"]).stem) for record in existing}
    known_paths = {record["image"] for record in existing}
    pending_paths = [path for path in paths if str(path) not in known_paths]
    maps = output / "maps"
    maps.mkdir(exist_ok=True)
    teacher = Teacher(saliency_model, face_model, config, threads, provider)
    created = 0

    def predict(path: Path) -> tuple[Path, str, str, np.ndarray, dict]:
        image = open_image(path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        target, metadata = teacher.predict(image)
        return path, digest, sha256_file(path), target, metadata

    with manifest.open("a") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = iter(pending_paths)
            in_flight = {}

            def fill() -> None:
                while len(in_flight) < workers:
                    try:
                        path = next(iterator)
                    except StopIteration:
                        return
                    in_flight[executor.submit(predict, path)] = path

            fill()
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    path, digest, image_sha256, target, metadata = future.result()
                    # Canonical decoded pixels also group identical PNG/WebP copies.
                    if digest not in known:
                        map_path = maps / f"{digest}.npy"
                        np.save(map_path, target, allow_pickle=False)
                        record = {
                            "image": os.path.relpath(path, output.resolve()),
                            "importance": f"maps/{digest}.npy",
                            "group": digest,
                            "image_id": digest,
                            "image_sha256": image_sha256,
                            "weight": 1.0,
                            **metadata,
                        }
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        known.add(digest)
                        created += 1
                        if created % 100 == 0 or len(known) == len(paths):
                            print(
                                f"Labeled {len(known):,}/{len(paths):,} images "
                                f"({created:,} created this run)",
                                flush=True,
                            )
                fill()
    return {"created": created, "total": len(known), "manifest": str(manifest)}
