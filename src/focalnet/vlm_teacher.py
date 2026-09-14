"""OpenRouter vision teacher: structured regions rasterized to 64×64 maps.

The U²-Net + YuNet teacher puts most mass on faces. This teacher asks a VLM
what the photograph is of, then writes the same oriented-image map contract
used by training. Inference stays a local ONNX student.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from focalnet.crop import Crop, clean_map, solve_crop
from focalnet.data import read_manifest, sha256_file
from focalnet.imaging import MAP_SIZE, image_paths, open_image

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
BAKEOFF_MODELS = (
    "google/gemini-3.1-flash-lite",
    "openai/gpt-5.6-luna",
    "anthropic/claude-haiku-4.5",
)
KINDS = frozenset({"face", "person", "animal", "text", "product", "object", "other"})
REGION_PROMPT = """\
Return JSON only, no markdown. Coordinates MUST be floats in [0, 1] \
relative to the full image (not pixels, not 0-100, not 0-1000). \
x1,y1 is the top-left and x2,y2 is the bottom-right.

Describe what a content-aware crop should keep, not just faces.
- Action, sports, dance, and jumping: include a high-importance box on the \
peak of the action (ball, rim, hands, product) AND a box on the full pose. \
Do not crop to the head or a centered torso.
- Multiple related subjects (person and dog, two people together) share a group.
- Background people, empty seats, and scenery get low importance.

Schema:
{
  "subjects": [
    {
      "label": "short name",
      "importance": 0.0,
      "kind": "face|person|animal|text|product|object|other",
      "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0
    }
  ],
  "keep_together": [0],
  "must_contain": [0]
}

importance is how much a crop must keep that subject (0-1).
keep_together and must_contain are 0-based indices into subjects.
must_contain lists boxes a valid crop may not clip.
keep_together must be a subset of must_contain. Do not include background
spectators unless they are the subject.
"""


@dataclass(frozen=True)
class Subject:
    label: str
    importance: float
    kind: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class VlmAnnotation:
    subjects: tuple[Subject, ...]
    keep_together: tuple[int, ...]
    must_contain: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "subjects": [asdict(subject) for subject in self.subjects],
            "keep_together": list(self.keep_together),
            "must_contain": list(self.must_contain),
        }

    def required_boxes(self) -> tuple[Subject, ...]:
        indexes = self.must_contain or self.keep_together
        if indexes:
            return tuple(self.subjects[index] for index in indexes)
        return tuple(subject for subject in self.subjects if subject.importance >= 0.5)


def parse_json_content(text: str) -> dict:
    content = text.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        content = content.removeprefix("json").strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model reply did not contain a JSON object")
    payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model JSON must be an object")
    return payload


def _index_list(values: object, count: int, field: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list of subject indexes")
    indexes = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must contain integer indexes")
        if not 0 <= value < count:
            raise ValueError(f"{field} index {value} is outside 0..{count - 1}")
        indexes.append(value)
    return tuple(indexes)


def _normalize_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    image_size: tuple[int, int] | None = None,
) -> tuple[float, float, float, float]:
    values = [x1, y1, x2, y2]
    if not np.isfinite(values).all():
        raise ValueError("Subject box must be finite")
    peak = max(abs(value) for value in values)
    if peak > 1:
        if peak <= 100:
            x1, y1, x2, y2 = x1 / 100, y1 / 100, x2 / 100, y2 / 100
        elif peak <= 1000:
            x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
        elif image_size is not None and min(image_size) > 0:
            width, height = image_size
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        else:
            raise ValueError("Subject box must be normalized with x1<x2 and y1<y2")
    x1, y1, x2, y2 = (float(np.clip(value, 0, 1)) for value in (x1, y1, x2, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise ValueError("Subject box must be normalized with x1<x2 and y1<y2")
    return x1, y1, x2, y2


def parse_annotation(
    payload: dict, *, image_size: tuple[int, int] | None = None
) -> VlmAnnotation:
    raw = payload.get("subjects")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Annotation needs a nonempty subjects list")
    subjects = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each subject must be an object")
        try:
            box = [float(item[name]) for name in ("x1", "y1", "x2", "y2")]
            importance = float(item.get("importance", 0))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Subject needs finite x1,y1,x2,y2 and importance") from exc
        if not np.isfinite(importance):
            raise ValueError("Subject needs finite x1,y1,x2,y2 and importance")
        x1, y1, x2, y2 = _normalize_box(*box, image_size=image_size)
        if not 0 <= importance <= 1:
            raise ValueError("Subject importance must be in [0, 1]")
        kind = str(item.get("kind", "other")).strip().lower() or "other"
        if kind not in KINDS:
            kind = "other"
        label = str(item.get("label", kind)).strip() or kind
        subjects.append(Subject(label, importance, kind, x1, y1, x2, y2))
    count = len(subjects)
    return VlmAnnotation(
        tuple(subjects),
        _index_list(payload.get("keep_together"), count, "keep_together"),
        _index_list(payload.get("must_contain"), count, "must_contain"),
    )


def region_importance_target(annotation: VlmAnnotation, size: int = MAP_SIZE) -> np.ndarray:
    """Fill subject boxes (plus a light Gaussian) and peak-scale to one.

    A Gaussian on a tall action box still peaks on the torso, so a zoomed crop
    can keep almost all mass. A filled must-contain box spreads mass to the
    extremities, so tight crops lose retention and the pose stays in frame.
    """
    if size <= 0:
        raise ValueError("Map size must be positive")
    heatmap = np.zeros((size, size), np.float64)
    yy, xx = np.meshgrid(
        (np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size, indexing="ij"
    )
    required = set(annotation.must_contain)
    for index, subject in enumerate(annotation.subjects):
        importance = subject.importance
        if required and index not in required:
            importance = min(importance, 0.2)
        if importance <= 0:
            continue
        filled = (
            (xx >= subject.x1) & (xx <= subject.x2) & (yy >= subject.y1) & (yy <= subject.y2)
        ).astype(np.float64)
        sigma_x = max((subject.x2 - subject.x1) / 2, 0.75 / size)
        sigma_y = max((subject.y2 - subject.y1) / 2, 0.75 / size)
        gaussian = np.exp(
            -0.5
            * (
                ((xx - (subject.x1 + subject.x2) / 2) / sigma_x) ** 2
                + ((yy - (subject.y1 + subject.y2) / 2) / sigma_y) ** 2
            )
        )
        region = filled + 0.25 * gaussian
        heatmap += region / region.sum() * importance
    peak = heatmap.max()
    return (heatmap / peak if peak else heatmap).astype(np.float32)


def box_coverage(
    crop: Crop, box: tuple[float, float, float, float], width: int, height: int
) -> float:
    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area == 0:
        return 1.0
    left, top = crop.left, crop.top
    overlap = max(0.0, min(left + crop.width, x2) - max(left, x1)) * max(
        0.0, min(top + crop.height, y2) - max(top, y1)
    )
    return float(overlap / area)


def encode_data_url(image: Image.Image, *, max_side: int = 1024, quality: int = 85) -> str:
    if min(image.size) <= 0:
        raise ValueError("Image dimensions must be positive")
    if max_side <= 0:
        raise ValueError("Max side must be positive")
    rgb = Image.new("RGB", image.size, (0, 0, 0))
    rgb.paste(image, mask=image.getchannel("A") if image.mode == "RGBA" else None)
    longest = max(rgb.size)
    if longest > max_side:
        scale = max_side / longest
        rgb = rgb.resize(
            (max(1, int(rgb.width * scale + 0.5)), max(1, int(rgb.height * scale + 0.5))),
            Image.Resampling.BILINEAR,
        )
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def openrouter_headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        raise ValueError("OpenRouter API key is empty")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/appwrite/focalnet",
        "X-Title": "FocalNet",
    }


def read_api_key(explicit: str | None = None) -> str:
    key = explicit if explicit is not None else os.environ.get("OPENROUTER_API_KEY", "")
    if not str(key).strip():
        raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")
    return str(key).strip()


class OpenRouterTeacher:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = 60,
        retries: int = 3,
        client: httpx.Client | None = None,
    ):
        if timeout <= 0 or retries <= 0:
            raise ValueError("Timeout and retries must be positive")
        self.api_key = read_api_key(api_key)
        self.model = model.strip()
        if not self.model:
            raise ValueError("Model id is empty")
        self.timeout = timeout
        self.retries = retries
        self.client = client

    def _payload(self, image: Image.Image, *, json_object: bool) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_data_url(image)}},
                        {"type": "text", "text": REGION_PROMPT},
                    ],
                }
            ],
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, image: Image.Image) -> tuple[VlmAnnotation, dict]:
        payload = self._payload(image, json_object=True)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._post(payload)
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") if isinstance(body, dict) else None
                if not choices:
                    raise ValueError("OpenRouter response is missing choices")
                message = choices[0].get("message") or {}
                annotation = parse_annotation(
                    parse_json_content(str(message.get("content") or "")),
                    image_size=image.size,
                )
                usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
                return annotation, {"model": self.model, "usage": usage}
            except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 400 and payload.get("response_format"):
                    payload = self._payload(image, json_object=False)
                    continue
                retryable = status in {429, 500, 502, 503, 504} or isinstance(
                    exc, httpx.TimeoutException
                )
                if not retryable:
                    break
                time.sleep(min(8.0, 0.5 * 2**attempt))
        raise ValueError(
            f"OpenRouter labeling failed for {self.model}: {last_error}"
        ) from last_error

    def _post(self, payload: dict) -> httpx.Response:
        if self.client is not None:
            return self.client.post(
                OPENROUTER_URL, json=payload, headers=openrouter_headers(self.api_key)
            )
        with httpx.Client(timeout=self.timeout, headers=openrouter_headers(self.api_key)) as client:
            return client.post(OPENROUTER_URL, json=payload)


def label_vlm_images(
    images: Path,
    output: Path,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    workers: int = 1,
    resume: bool = False,
    teacher: OpenRouterTeacher | None = None,
) -> dict:
    if workers <= 0:
        raise ValueError("Label workers must be positive")
    paths = image_paths(images)
    teacher = teacher or OpenRouterTeacher(read_api_key(api_key), model)
    provenance = {
        "format_version": 1,
        "target": "vlm-region-mixture-v1",
        "layout": "oriented-image",
        "map_size": MAP_SIZE,
        "teacher": "openrouter",
        "model": teacher.model,
        "prompt_sha256": hashlib.sha256(REGION_PROMPT.encode()).hexdigest(),
    }
    manifest = output / "labels.jsonl"
    if resume:
        if json.loads((output / "provenance.json").read_text()) != provenance:
            raise ValueError("Resume requires identical VLM teacher settings")
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
    pending = [path for path in paths if str(path) not in known_paths]
    maps = output / "maps"
    maps.mkdir(exist_ok=True)
    created = 0

    def predict(path: Path) -> tuple[Path, str, str, np.ndarray, dict]:
        image = open_image(path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        annotation, metadata = teacher.complete(image)
        target = region_importance_target(annotation)
        return (
            path,
            digest,
            sha256_file(path),
            target,
            {**metadata, "annotation": annotation.to_dict()},
        )

    with manifest.open("a") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = iter(pending)
            in_flight: dict = {}

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
                    if digest not in known:
                        np.save(maps / f"{digest}.npy", target, allow_pickle=False)
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
                        print(
                            f"Labeled {len(known):,}/{len(paths):,} images "
                            f"({created:,} created this run)",
                            flush=True,
                        )
                fill()
    return {
        "created": created,
        "total": len(known),
        "manifest": str(manifest),
        "model": teacher.model,
    }


def crop_containment(
    heatmap: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    ratios: list[float],
) -> dict[str, dict]:
    values = clean_map(heatmap)
    results = {}
    for ratio in ratios:
        crop = solve_crop(values, width, height, ratio)
        coverages = [box_coverage(crop, box, width, height) for box in boxes]
        results[str(ratio)] = {
            "min_box_coverage": float(min(coverages) if coverages else 1.0),
            "mean_box_coverage": float(np.mean(coverages) if coverages else 1.0),
            **crop.to_dict(),
        }
    return results


def bakeoff_vlm(
    images: Path,
    models: list[str],
    *,
    api_key: str | None = None,
    ratios: list[float] | None = None,
    expected: dict[str, list[tuple[float, float, float, float]]] | None = None,
    teacher_factory=None,
) -> dict:
    paths = image_paths(images)
    if not models:
        raise ValueError("Provide at least one OpenRouter model id")
    ratios = ratios or [1.0, 4 / 5, 9 / 16]
    key = None if teacher_factory else read_api_key(api_key)
    per_model = []
    for model in models:
        teacher = teacher_factory(model) if teacher_factory else OpenRouterTeacher(key, model)
        records = []
        for path in paths:
            image = open_image(path)
            annotation, metadata = teacher.complete(image)
            heatmap = region_importance_target(annotation)
            boxes = expected.get(path.name) if expected else None
            if not boxes:
                boxes = [subject.box for subject in annotation.required_boxes()]
            records.append(
                {
                    "image": path.name,
                    "annotation": annotation.to_dict(),
                    "crops": crop_containment(heatmap, boxes, image.width, image.height, ratios),
                    "usage": metadata.get("usage", {}),
                }
            )
        coverages = [
            record["crops"][str(ratio)]["min_box_coverage"]
            for record in records
            for ratio in ratios
        ]
        per_model.append(
            {
                "model": model,
                "mean_min_box_coverage": float(np.mean(coverages) if coverages else 0.0),
                "images": records,
            }
        )
    ranking = sorted(per_model, key=lambda item: item["mean_min_box_coverage"], reverse=True)
    return {
        "reference": "VLM region maps: min coverage of must-contain boxes after solve_crop",
        "ratios": [str(ratio) for ratio in ratios],
        "models": ranking,
    }


def load_expected_boxes(path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Expected-boxes file must be a JSON object keyed by filename")
    expected = {}
    for name, boxes in payload.items():
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{name}: expected a nonempty list of boxes")
        parsed = []
        for box in boxes:
            if isinstance(box, dict):
                values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
            else:
                values = [float(value) for value in box]
            if len(values) != 4 or not 0 <= values[0] < values[2] <= 1:
                raise ValueError(f"{name}: boxes must be normalized x1,y1,x2,y2")
            if not 0 <= values[1] < values[3] <= 1:
                raise ValueError(f"{name}: boxes must be normalized x1,y1,x2,y2")
            parsed.append(tuple(values))
        expected[str(name)] = parsed
    return expected
