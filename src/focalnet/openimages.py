"""Resumable, attribution-preserving Open Images V7 subset acquisition."""

import csv
import hashlib
import json
import math
import random
import re
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from io import BytesIO, TextIOWrapper
from pathlib import Path
from urllib.request import Request, urlopen

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
BOX_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
IMAGE_ID = re.compile(r"^[0-9a-f]{16}$")
MIB = 1024**2
GIB = 1024**3

CATEGORY_NAMES = {
    "human": {
        "Person",
        "Man",
        "Woman",
        "Boy",
        "Girl",
        "Human face",
        "Human head",
    },
    "animal": {
        "Animal",
        "Bird",
        "Cat",
        "Dog",
        "Horse",
        "Fish",
        "Cattle",
        "Sheep",
        "Goat",
        "Elephant",
        "Bear",
        "Zebra",
        "Giraffe",
        "Monkey",
        "Rabbit",
        "Tiger",
        "Lion",
        "Reptile",
        "Insect",
        "Butterfly",
        "Duck",
        "Chicken",
        "Pig",
        "Deer",
        "Camel",
    },
    "food": {
        "Food",
        "Fruit",
        "Vegetable",
        "Fast food",
        "Baked goods",
        "Dessert",
        "Bread",
        "Cake",
        "Pizza",
        "Sandwich",
        "Salad",
        "Apple",
        "Banana",
        "Orange",
        "Tomato",
        "Strawberry",
        "Drink",
        "Wine",
        "Coffee",
        "Beer",
        "Plate",
        "Bowl",
    },
    "vehicle": {
        "Vehicle",
        "Car",
        "Truck",
        "Bus",
        "Train",
        "Airplane",
        "Boat",
        "Bicycle",
        "Motorcycle",
        "Van",
        "Taxi",
        "Helicopter",
        "Ship",
        "Ambulance",
        "Cart",
    },
    "scene": {
        "Building",
        "House",
        "Skyscraper",
        "Tower",
        "Castle",
        "Tree",
        "Plant",
        "Flower",
        "Palm tree",
        "Houseplant",
        "Fountain",
        "Bridge",
        "Tent",
        "Sculpture",
        "Bench",
    },
}


@dataclass(frozen=True)
class AcquireConfig:
    count: int = 100_000
    max_side: int = 640
    workers: int = 32
    storage_limit_gib: float = 75
    min_free_gib: float = 35
    seed: int = 42
    jpeg_quality: int = 88
    mix: str = "default"

    def __post_init__(self):
        if min(self.count, self.max_side, self.workers) <= 0:
            raise ValueError("Count, max-side, and workers must be positive")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("JPEG quality must be between 1 and 95")
        if self.mix not in {"default", "expanded"}:
            raise ValueError("Mix must be default or expanded")
        if not math.isfinite(self.storage_limit_gib) or self.storage_limit_gib <= 0:
            raise ValueError("Storage limit must be finite and positive")
        if not math.isfinite(self.min_free_gib) or self.min_free_gib < 1:
            raise ValueError("Minimum free space must be finite and at least 1 GiB")


def _quotas(count: int, mix: str = "default") -> dict[str, int]:
    weights = (
        {
            "human": 0.30,
            "animal": 0.15,
            "food": 0.15,
            "vehicle": 0.15,
            "scene": 0.10,
            "busy": 0.10,
        }
        if mix == "default"
        else {
            # Open Images cannot supply the default food/busy proportions at 400k scale.
            "human": 0.375,
            "animal": 0.1625,
            "food": 0.11,
            "vehicle": 0.1625,
            "scene": 0.11,
            "busy": 0.04,
        }
    )
    quotas = {name: int(count * weight) for name, weight in weights.items()}
    quotas["general"] = count - sum(quotas.values())
    return quotas


def _open_csv(source: str):
    response = urlopen(Request(source, headers={"User-Agent": "focalnet/0.1"}), timeout=120)
    return response, csv.reader(TextIOWrapper(response, encoding="utf-8", newline=""))


def _class_ids(source: str = CLASS_URL) -> dict[str, str]:
    response, reader = _open_csv(source)
    try:
        names = {name: label for label, name in reader}
    finally:
        response.close()
    return {
        label: bucket
        for bucket, wanted in CATEGORY_NAMES.items()
        for name in wanted
        if (label := names.get(name)) is not None
    }


class _Reservoir:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.values: list[dict] = []

    def add(self, value: dict) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.values[index] = value


def _choose_bucket(labels: set[str], box_count: int, mapping: dict[str, str]) -> str:
    present = {mapping[label] for label in labels if label in mapping}
    for bucket in ("human", "animal", "food", "vehicle", "scene"):
        if bucket in present:
            return bucket
    return "busy" if box_count >= 6 else "general"


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan_candidates(
    destination: Path,
    config: AcquireConfig,
    *,
    class_url: str = CLASS_URL,
    box_url: str = BOX_URL,
    exclude_image_ids: set[str] | None = None,
) -> dict:
    exclude_image_ids = exclude_image_ids or set()
    quotas = _quotas(config.count, config.mix)
    mapping = _class_ids(class_url)
    if not mapping:
        raise ValueError("No requested category names exist in the Open Images class table")
    rng = random.Random(config.seed)
    reservoirs = {
        bucket: _Reservoir(max(quota + 100, int(quota * 1.25)), rng)
        for bucket, quota in quotas.items()
    }
    response, reader = _open_csv(box_url)
    try:
        header = next(reader)
        try:
            image_index, label_index = header.index("ImageID"), header.index("LabelName")
        except ValueError as exc:
            raise ValueError("Unexpected Open Images bounding-box columns") from exc
        current, labels, boxes, images = None, set(), 0, 0

        def finish() -> None:
            nonlocal images
            if current is None:
                return
            images += 1
            if current in exclude_image_ids:
                return
            bucket = _choose_bucket(labels, boxes, mapping)
            reservoirs[bucket].add(
                {
                    "image_id": current,
                    "bucket": bucket,
                    "box_count": boxes,
                    "labels": sorted(labels),
                }
            )
            if images % 100_000 == 0:
                print(f"Scanned {images:,} annotated images", flush=True)

        for row in reader:
            image_id = row[image_index].lower()
            if image_id != current:
                finish()
                current, labels, boxes = image_id, set(), 0
            labels.add(row[label_index])
            boxes += 1
        finish()
    finally:
        response.close()
    missing = {
        bucket: quotas[bucket] - len(pool.values)
        for bucket, pool in reservoirs.items()
        if len(pool.values) < quotas[bucket]
    }
    if missing:
        raise ValueError(f"Open Images could not satisfy category quotas: {missing}")
    temporary = destination.with_suffix(".jsonl.tmp")
    counts = {}
    with temporary.open("x") as handle:
        for bucket, pool in reservoirs.items():
            rng.shuffle(pool.values)
            counts[bucket] = len(pool.values)
            for record in pool.values:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    return {"scanned": images, "candidates": sum(counts.values()), "by_bucket": counts}


def attach_metadata(
    candidates_path: Path, destination: Path, *, metadata_url: str = METADATA_URL
) -> dict:
    candidates = {}
    for line in candidates_path.read_text().splitlines():
        record = json.loads(line)
        candidates[record["image_id"]] = record
    response, reader = _open_csv(metadata_url)
    found = 0
    temporary = destination.with_suffix(".jsonl.tmp")
    try:
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        required = [
            "ImageID",
            "OriginalURL",
            "OriginalLandingURL",
            "License",
            "Author",
            "AuthorProfileURL",
            "Title",
            "Rotation",
        ]
        if not set(required) <= index.keys():
            raise ValueError("Unexpected Open Images metadata columns")
        with temporary.open("x") as handle:
            for row in reader:
                image_id = row[index["ImageID"]].lower()
                candidate = candidates.get(image_id)
                if candidate is None:
                    continue
                metadata = {name: row[index[name]] for name in required[1:]}
                if "creativecommons.org/licenses/by/2.0" not in metadata["License"]:
                    continue
                handle.write(json.dumps({**candidate, **metadata}, separators=(",", ":")) + "\n")
                found += 1
    finally:
        response.close()
    temporary.replace(destination)
    return {"requested": len(candidates), "with_cc_by_metadata": found}


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _image_bytes(
    image_id: str, max_bytes: int = 40 * MIB, client: httpx.Client | None = None
) -> bytes:
    url = IMAGE_URL.format(image_id=image_id)
    last_error = None
    for attempt in range(3):
        try:
            if client is not None:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_length = int(response.headers.get("Content-Length", 0))
                    if content_length > max_bytes:
                        raise ValueError(f"source image exceeds {max_bytes // MIB} MiB")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes(chunk_size=MIB):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"source image exceeds {max_bytes // MIB} MiB")
                        chunks.append(chunk)
                    return b"".join(chunks)
            with urlopen(
                Request(url, headers={"User-Agent": "focalnet/0.1"}), timeout=60
            ) as response:
                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > max_bytes:
                    raise ValueError(f"source image exceeds {max_bytes // MIB} MiB")
                chunks, size = [], 0
                while chunk := response.read(MIB):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"source image exceeds {max_bytes // MIB} MiB")
                    chunks.append(chunk)
                return b"".join(chunks)
        except Exception as exc:  # network failures vary across urllib and SSL implementations
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def _normalise_jpeg(
    data: bytes, rotation: str, max_side: int, quality: int
) -> tuple[bytes, int, int]:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > 100_000_000:
                raise ValueError("decoded image exceeds 100 megapixels")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(f"invalid source image: {exc}") from exc
    try:
        degrees = 0 if not rotation or rotation.lower() == "nan" else int(float(rotation))
    except ValueError as exc:
        raise ValueError(f"invalid rotation: {rotation}") from exc
    if degrees not in (0, 90, 180, 270):
        raise ValueError(f"invalid rotation: {rotation}")
    if degrees:
        image = image.rotate(degrees, expand=True)
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    return output.getvalue(), image.width, image.height


def _download_one(
    record: dict, images: Path, config: AcquireConfig, client: httpx.Client | None = None
) -> dict:
    image_id = record["image_id"]
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError(f"invalid Open Images ID: {image_id}")
    relative = Path(image_id[:2]) / f"{image_id}.jpg"
    target = images / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    new_bytes = 0
    if target.exists():
        data = target.read_bytes()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    else:
        data, width, height = _normalise_jpeg(
            _image_bytes(image_id, client=client),
            record["Rotation"],
            config.max_side,
            config.jpeg_quality,
        )
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{image_id}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        temporary_path.replace(target)
        new_bytes = len(data)
    return {
        **record,
        "image": str(Path("images") / relative),
        "width": width,
        "height": height,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "_new_bytes": new_bytes,
    }


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def download_candidates(output: Path, config: AcquireConfig) -> dict:
    candidates = defaultdict(list)
    for record in _load_jsonl(output / "candidates-with-metadata.jsonl"):
        candidates[record["bucket"]].append(record)
    rng = random.Random(config.seed + 1)
    for records in candidates.values():
        rng.shuffle(records)
    manifest = output / "images.jsonl"
    completed_records = _load_jsonl(manifest)
    completed = {record["image_id"] for record in completed_records}
    counts = defaultdict(int)
    for record in completed_records:
        counts[record["bucket"]] += 1
    quotas = _quotas(config.count, config.mix)
    errors = output / "failures.jsonl"
    images = output / "images"
    images.mkdir(exist_ok=True)
    used = _directory_size(output)
    limit = int(config.storage_limit_gib * GIB)
    floor = int(config.min_free_gib * GIB)
    limits = httpx.Limits(
        max_connections=config.workers,
        max_keepalive_connections=config.workers,
        keepalive_expiry=30,
    )
    with manifest.open("a") as success_handle, errors.open("a") as error_handle:
        with (
            httpx.Client(
                headers={"User-Agent": "focalnet/0.1"},
                timeout=60,
                limits=limits,
                follow_redirects=True,
            ) as client,
            ThreadPoolExecutor(max_workers=config.workers) as executor,
        ):
            for bucket, quota in quotas.items():
                pool = [
                    record for record in candidates[bucket] if record["image_id"] not in completed
                ]
                cursor = 0
                in_flight = {}
                while counts[bucket] < quota:
                    if used >= limit:
                        raise RuntimeError(
                            f"dataset reached {config.storage_limit_gib:g} GiB limit"
                        )
                    free = shutil.disk_usage(output).free
                    if free < floor:
                        raise RuntimeError(
                            f"free disk space fell below {config.min_free_gib:g} GiB"
                        )
                    needed = quota - counts[bucket]
                    while len(in_flight) < min(needed, config.workers) and cursor < len(pool):
                        record = pool[cursor]
                        cursor += 1
                        future = executor.submit(_download_one, record, images, config, client)
                        in_flight[future] = record
                    if not in_flight:
                        raise RuntimeError(f"candidate pool exhausted for {bucket}")
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        record = in_flight.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            error_handle.write(
                                json.dumps(
                                    {
                                        "image_id": record["image_id"],
                                        "bucket": bucket,
                                        "error": str(exc),
                                    }
                                )
                                + "\n"
                            )
                            error_handle.flush()
                            continue
                        new_bytes = result.pop("_new_bytes")
                        success_handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                        success_handle.flush()
                        completed.add(result["image_id"])
                        counts[bucket] += 1
                        used += new_bytes
                        total = sum(counts.values())
                        if total % 1000 == 0:
                            print(
                                f"Downloaded {total:,}/{config.count:,} images "
                                f"({used / GIB:.1f} GiB dataset, {free / GIB:.1f} GiB free)",
                                flush=True,
                            )
    return {
        "downloaded": sum(counts.values()),
        "by_bucket": dict(counts),
        "dataset_gib": _directory_size(output) / GIB,
        "free_gib": shutil.disk_usage(output).free / GIB,
    }


def acquire_open_images(
    output: Path,
    config: AcquireConfig,
    exclude_manifest: Path | None = None,
    *,
    plan_only: bool = False,
) -> dict:
    excluded = set()
    exclusion = None
    if exclude_manifest is not None:
        exclude_manifest = exclude_manifest.resolve()
        for number, line in enumerate(exclude_manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            image_id = str(json.loads(line).get("image_id", "")).lower()
            if not IMAGE_ID.fullmatch(image_id):
                raise ValueError(f"Invalid image_id in {exclude_manifest}:{number}")
            excluded.add(image_id)
        exclusion = {
            "images": len(excluded),
            "sha256": hashlib.sha256(exclude_manifest.read_bytes()).hexdigest(),
        }
    provenance = {
        "format_version": 1,
        "dataset": "Open Images V7 train subset",
        "config": asdict(config),
        "quotas": _quotas(config.count, config.mix),
        "sources": {
            "classes": CLASS_URL,
            "boxes": BOX_URL,
            "metadata": METADATA_URL,
            "images": IMAGE_URL,
        },
        "license_policy": "retain only records declaring CC BY 2.0 metadata",
        "exclusion": exclusion,
    }
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text())
        immutable_settings = ("count", "max_side", "seed", "jpeg_quality")
        same_dataset = all(
            existing.get(field) == provenance[field]
            for field in (
                "format_version",
                "dataset",
                "quotas",
                "sources",
                "license_policy",
                "exclusion",
            )
        )
        same_output = all(
            existing.get("config", {}).get(field) == provenance["config"][field]
            for field in immutable_settings
        )
        if not same_dataset or not same_output:
            raise ValueError(
                "Existing acquisition uses different dataset-output settings; "
                "choose a new output path"
            )
    else:
        _atomic_json(provenance_path, provenance)
    candidates = output / "candidates.jsonl"
    if not candidates.exists():
        print("Planning a balanced subset from Open Images bounding boxes…", flush=True)
        print(
            json.dumps(plan_candidates(candidates, config, exclude_image_ids=excluded)), flush=True
        )
    enriched = output / "candidates-with-metadata.jsonl"
    if not enriched.exists():
        print("Attaching attribution, license, and rotation metadata…", flush=True)
        report = attach_metadata(candidates, enriched)
        print(json.dumps(report), flush=True)
        available = defaultdict(int)
        for record in _load_jsonl(enriched):
            available[record["bucket"]] += 1
        missing = {
            bucket: quota - available[bucket]
            for bucket, quota in _quotas(config.count, config.mix).items()
            if available[bucket] < quota
        }
        if missing:
            raise ValueError(f"Not enough candidates with retained CC BY metadata: {missing}")
    if plan_only:
        available = defaultdict(int)
        for record in _load_jsonl(enriched):
            available[record["bucket"]] += 1
        return {
            "planned_candidates": sum(available.values()),
            "by_bucket": dict(available),
            "excluded": len(excluded),
        }
    result = download_candidates(output, config)
    _atomic_json(
        output / "acquisition-report.json",
        {
            **result,
            "workers": config.workers,
            "storage_limit_gib": config.storage_limit_gib,
            "min_free_gib": config.min_free_gib,
        },
    )
    return result
