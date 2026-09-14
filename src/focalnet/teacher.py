"""Independent ONNX teacher using Autogravity's preprocessing and YuNet rules.

The target differs intentionally from Autogravity's single-face short circuit:
both models run, and every retained face contributes to the importance map.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from focalnet.crop import clean_map
from focalnet.imaging import MAP_SIZE, Letterbox, prepare_image, restore_map


@dataclass(frozen=True)
class Face:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def priority(self) -> float:
        return self.confidence * math.sqrt((self.x2 - self.x1) * (self.y2 - self.y1))


@dataclass(frozen=True)
class TeacherConfig:
    face_weight: float = 3.0
    score_threshold: float = 0.85
    nms_threshold: float = 0.30
    top_k: int = 5000

    def __post_init__(self):
        if not math.isfinite(self.face_weight) or self.face_weight < 0:
            raise ValueError("Face weight must be finite and nonnegative")
        if not 0 < self.score_threshold <= 1 or not 0 < self.nms_threshold <= 1:
            raise ValueError("Detection thresholds must be in (0, 1]")
        if self.top_k <= 0:
            raise ValueError("Top-k must be positive")


def cpu_session(path: str | Path, threads: int = 1) -> ort.InferenceSession:
    if threads <= 0:
        raise ValueError("Threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # ORT's CPU graph fusions can cause platform-dependent probability drift for trained RepViT
    # graphs. Keep the verified runtime contract architecture-independent.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def teacher_session(
    path: str | Path, threads: int = 1, provider: str = "cpu"
) -> ort.InferenceSession:
    if provider == "cpu":
        return cpu_session(path, threads)
    providers = {
        "coreml": "CoreMLExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    if provider not in providers:
        raise ValueError("Teacher provider must be cpu, coreml, or cuda")
    execution_provider = providers[provider]
    if provider == "cuda":
        ort.preload_dlls(directory="")
    if execution_provider not in ort.get_available_providers():
        raise ValueError(f"{provider.upper()} execution provider is unavailable")
    if threads <= 0:
        raise ValueError("Threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path), options, providers=[execution_provider, "CPUExecutionProvider"]
    )
    if execution_provider not in session.get_providers():
        raise ValueError(f"{provider.upper()} execution provider failed to initialize")
    return session


def decode_faces(
    outputs: dict[str, np.ndarray], box: Letterbox, config: TeacherConfig
) -> list[Face]:
    candidates = []
    for stride in (8, 16, 32):
        columns = box.size // stride
        cls = np.clip(outputs[f"cls_{stride}"].reshape(-1).astype(np.float64), 0, 1)
        obj = np.clip(outputs[f"obj_{stride}"].reshape(-1).astype(np.float64), 0, 1)
        boxes = outputs[f"bbox_{stride}"].reshape(-1, 4).astype(np.float64)
        if cls.shape != (columns * columns,) or obj.shape != cls.shape or len(boxes) != len(cls):
            raise ValueError(f"Unexpected YuNet output shape at stride {stride}")
        scores = np.sqrt(cls * obj)
        indices = np.flatnonzero(np.isfinite(scores) & (scores >= config.score_threshold))
        selected = boxes[indices]
        centers = np.stack((indices % columns, indices // columns), axis=1)
        centers = (centers + selected[:, :2]) * stride
        with np.errstate(over="ignore", invalid="ignore"):
            sizes = np.exp(selected[:, 2:]) * stride
        corners = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        usable = np.isfinite(corners).all(axis=1) & (sizes > 0).all(axis=1)
        candidates.extend(np.column_stack((corners, scores[indices]))[usable])
    if not candidates:
        return []
    candidates = np.asarray(candidates)
    order = np.argsort(-candidates[:, 4], kind="stable")[: config.top_k]
    kept = []
    while len(order):
        index = order[0]
        kept.append(candidates[index])
        others = candidates[order[1:]]
        current = candidates[index]
        overlap = np.maximum(
            0, np.minimum(current[2:4], others[:, 2:4]) - np.maximum(current[:2], others[:, :2])
        ).prod(axis=1)
        areas = (others[:, 2:4] - others[:, :2]).prod(axis=1)
        area = (current[2:4] - current[:2]).prod()
        iou = overlap / np.maximum(area + areas - overlap, 1e-12)
        order = order[1:][iou < config.nms_threshold]
    faces = []
    for x1, y1, x2, y2, confidence in kept:
        x1, x2 = np.clip((np.array([x1, x2]) - box.left) / box.width, 0, 1)
        y1, y2 = np.clip((np.array([y1, y2]) - box.top) / box.height, 0, 1)
        if x2 > x1 and y2 > y1:
            faces.append(Face(float(x1), float(y1), float(x2), float(y2), float(confidence)))
    return faces


def importance_target(
    saliency: np.ndarray, faces: list[Face], config: TeacherConfig | None = None
) -> np.ndarray:
    """Mix saliency and face *mass*, then scale the peak to one.

    With face_weight=3 and both signals present, faces receive 75% of total
    mass. Boosting only Gaussian peaks would let large bodies overwhelm faces.
    Face mass is apportioned by confidence * sqrt(area), like Autogravity's
    prominence score. No-face images retain the saliency distribution.
    """
    config = config or TeacherConfig()
    saliency = np.clip(clean_map(saliency), 0, 1)
    height, width = saliency.shape
    yy, xx = np.meshgrid(
        (np.arange(height) + 0.5) / height, (np.arange(width) + 0.5) / width, indexing="ij"
    )
    face_map = np.zeros_like(saliency)
    for face in faces:
        values = [face.x1, face.y1, face.x2, face.y2, face.confidence]
        if (
            not np.isfinite(values).all()
            or not 0 <= face.x1 < face.x2 <= 1
            or not 0 <= face.y1 < face.y2 <= 1
            or not 0 < face.confidence <= 1
        ):
            raise ValueError("Face must have a finite normalized box and confidence in (0, 1]")
        if face.confidence < config.score_threshold:
            continue
        sigma_x = max((face.x2 - face.x1) / 2, 0.75 / width)
        sigma_y = max((face.y2 - face.y1) / 2, 0.75 / height)
        gaussian = np.exp(
            -0.5
            * (
                ((xx - (face.x1 + face.x2) / 2) / sigma_x) ** 2
                + ((yy - (face.y1 + face.y2) / 2) / sigma_y) ** 2
            )
        )
        face_map += gaussian / gaussian.sum() * face.priority
    if saliency.sum() > 0:
        saliency /= saliency.sum()
    if face_map.sum() > 0:
        saliency += config.face_weight * face_map / face_map.sum()
    peak = saliency.max()
    return (saliency / peak if peak else saliency).astype(np.float32)


class Teacher:
    def __init__(
        self,
        saliency_model: Path,
        face_model: Path,
        config: TeacherConfig | None = None,
        threads: int = 1,
        provider: str = "cpu",
    ):
        self.config = config or TeacherConfig()
        self.saliency = teacher_session(saliency_model, threads, provider)
        self.faces = teacher_session(face_model, threads, provider)
        self.face_output_names = [
            f"{name}_{stride}" for name in ("cls", "obj", "bbox") for stride in (8, 16, 32)
        ]
        if self.saliency.get_inputs()[0].shape != [1, 3, 320, 320]:
            raise ValueError("Teacher requires Autogravity's 320x320 U²-Net ONNX model")
        if self.faces.get_inputs()[0].shape != [1, 3, 640, 640]:
            raise ValueError("Teacher requires the fixed 640x640 YuNet 2023mar ONNX model")
        if "1959" not in {output.name for output in self.saliency.get_outputs()}:
            raise ValueError("U²-Net model is missing its fused output '1959'")
        if not set(self.face_output_names) <= {output.name for output in self.faces.get_outputs()}:
            raise ValueError("YuNet model has an incompatible output contract")

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict]:
        face_input, face_box = prepare_image(image, 640, bgr=True)
        raw_faces = self.faces.run(
            self.face_output_names, {self.faces.get_inputs()[0].name: face_input}
        )
        detections = decode_faces(
            dict(zip(self.face_output_names, raw_faces, strict=True)), face_box, self.config
        )
        saliency_input, saliency_box = prepare_image(image, 320)
        saliency = self.saliency.run(
            ["1959"], {self.saliency.get_inputs()[0].name: saliency_input}
        )[0][0, 0]
        if not np.isfinite(saliency).all():
            raise ValueError("Teacher produced nonfinite saliency")
        source_map = restore_map(saliency, saliency_box, MAP_SIZE)
        return importance_target(source_map, detections, self.config), {
            "faces": [asdict(face) for face in detections],
            "face_count": len(detections),
            "saliency_peak": float(source_map.max()),
        }
