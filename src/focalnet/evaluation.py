"""Measure teacher agreement, crop retention, and actual CPU latency."""

import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort

from focalnet.candidates import MAX_CANDIDATES, generate_candidates, letterbox_content
from focalnet.crop import focal_point, score_crop, solve_crop
from focalnet.data import read_manifest
from focalnet.human_runtime import HumanCropPredictor
from focalnet.imaging import image_paths, open_image, prepare_image
from focalnet.runtime import Predictor


def evaluate(model: Path, manifest: Path, ratios: list[float], threads: int = 1) -> dict:
    records = read_manifest(manifest)
    predictor = Predictor(model, threads)
    calibration = set(predictor.metadata.get("quantization", {}).get("groups", []))
    if calibration & {record["group"] for record in records}:
        raise ValueError("Evaluation groups overlap the model's INT8 calibration groups")
    results = []
    for record in records:
        image = open_image(record["image"])
        prediction = predictor.heatmap(image)
        target = np.load(record["importance"], allow_pickle=False)
        if (
            target.shape != prediction.shape
            or not np.isfinite(target).all()
            or target.min() < 0
            or target.max() > 1
        ):
            raise ValueError(f"Invalid target: {record['importance']}")
        point, reference = focal_point(prediction), focal_point(target)
        result = {
            "image": record["image"],
            "map_mae": float(np.abs(prediction - target).mean()),
            "centroid_error": float(
                np.hypot(point["x"] - reference["x"], point["y"] - reference["y"])
            ),
            "crops": {},
        }
        for ratio in ratios:
            crop = solve_crop(prediction, image.width, image.height, ratio)
            oracle = solve_crop(target, image.width, image.height, ratio)
            center = replace(
                crop, left=(image.width - crop.width) // 2, top=(image.height - crop.height) // 2
            )
            retained = score_crop(target, crop, image.width, image.height)
            result["crops"][str(ratio)] = {
                "retention": retained,
                "oracle_retention": oracle.retained_importance,
                "center_retention": score_crop(target, center, image.width, image.height),
                "regret": max(0.0, oracle.retained_importance - retained),
            }
        results.append(result)
    return {
        "images": len(results),
        "reference": "teacher importance maps (not human crop judgments)",
        "map_mae": float(np.mean([r["map_mae"] for r in results])),
        "centroid_error": float(np.mean([r["centroid_error"] for r in results])),
        "crops": {
            str(ratio): {
                metric: float(np.mean([r["crops"][str(ratio)][metric] for r in results]))
                for metric in ("retention", "oracle_retention", "center_retention", "regret")
            }
            for ratio in ratios
        },
        "per_image": results,
    }


def benchmark(
    model: Path, images: Path, *, iterations: int = 50, warmup: int = 5, threads: int = 1
) -> dict:
    if iterations <= 0 or warmup < 0:
        raise ValueError("Iterations must be positive and warmup nonnegative")
    paths = image_paths(images)
    predictor = Predictor(model, threads)
    tensor, _ = prepare_image(open_image(paths[0]))
    for _ in range(warmup):
        predictor.session.run(["importance"], {"image": tensor})
    forward, pipeline = [], []
    for index in range(iterations):
        start = time.perf_counter()
        predictor.session.run(["importance"], {"image": tensor})
        forward.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        predictor.predict(paths[index % len(paths)], 1.0)
        pipeline.append((time.perf_counter() - start) * 1000)

    def stats(values: list[float]) -> dict:
        return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "threads": threads,
        "iterations": iterations,
        "warmup": warmup,
        "model_bytes": model.stat().st_size,
        "forward": stats(forward),
        "pipeline": stats(pipeline),
        "scope": "Sequential batch=1. Forward uses one preprocessed image. Pipeline cycles images, "
        "including file read, decode, preprocessing, inference, focal point, and square crop. "
        "Model startup and HTTP overhead excluded.",
    }


def benchmark_human(
    model: Path,
    images: Path,
    *,
    aspect_ratio: float = 1.0,
    iterations: int = 50,
    warmup: int = 5,
    threads: int = 1,
) -> dict:
    if iterations <= 0 or warmup < 0:
        raise ValueError("Iterations must be positive and warmup nonnegative")
    paths = image_paths(images)
    predictor = HumanCropPredictor(model, threads)
    image = open_image(paths[0])
    tensor, content_box = prepare_image(image)
    candidates = generate_candidates(image.width, image.height, aspect_ratio)
    boxes = np.zeros((1, MAX_CANDIDATES, 4), dtype=np.float32)
    boxes[0, : len(candidates)] = candidates
    feeds = {
        "image": tensor,
        "boxes": boxes,
        "content": letterbox_content(content_box)[None],
    }
    for _ in range(warmup):
        predictor.session.run(["importance", "crop_scores"], feeds)
    forward, pipeline = [], []
    for index in range(iterations):
        start = time.perf_counter()
        predictor.session.run(["importance", "crop_scores"], feeds)
        forward.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        predictor.predict(paths[index % len(paths)], aspect_ratio)
        pipeline.append((time.perf_counter() - start) * 1000)

    def stats(values: list[float]) -> dict:
        return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "threads": threads,
        "iterations": iterations,
        "warmup": warmup,
        "aspect_ratio": aspect_ratio,
        "model_bytes": model.stat().st_size,
        "forward": stats(forward),
        "pipeline": stats(pipeline),
        "scope": "Sequential batch=1. Forward uses one preprocessed image and a padded crop "
        "candidate list. Pipeline cycles images, including file read, decode, preprocessing, "
        "candidate generation, inference, importance-retention safety gate, and crop selection. "
        "Model startup and HTTP overhead excluded.",
    }
