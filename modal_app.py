"""Train FocalNet on a Modal GPU with data stored in a persistent Volume."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

APP_NAME = "appwrite-focalnet"
VOLUME_NAME = "focalnet-data"
DATA_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=1.26,<3",
        "pillow>=11,<13",
        "httpx>=0.28,<1",
        "torch>=2.6,<3",
        "timm>=1.0.15,<2",
        "onnx>=1.17,<2",
        "onnxscript>=0.3,<1",
        "onnxruntime>=1.23.2,<2",
    )
    .add_local_dir("src", "/opt/focalnet/src")
)
teacher_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=1.26,<3",
        "pillow>=11,<13",
        "httpx>=0.28,<1",
        "onnxruntime-gpu[cuda,cudnn]==1.23.2",
    )
    .add_local_dir("src", "/opt/focalnet/src")
)
cpc_download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("gdown>=5,<6")


def _import_project() -> None:
    sys.path.insert(0, "/opt/focalnet/src")


def _planned_image_paths(dataset: Path, expected: int) -> list[Path]:
    """Read exact normalized image paths without traversing a large Volume tree."""
    import json

    manifest = dataset / "images.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing normalized image manifest: {manifest}")
    root = (dataset / "images").absolute()
    paths = []
    for number, line in enumerate(manifest.open(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        relative = Path(record["image"])
        path = (dataset / relative).absolute()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Image path escapes dataset at {manifest}:{number}")
        paths.append(path)
    if len(paths) != expected or len(set(paths)) != expected:
        raise ValueError(f"Expected {expected:,} unique normalized image paths")
    return sorted(paths)


@app.function(
    image=cpc_download_image,
    volumes={"/data": volume},
    cpu=2,
    memory=4096,
    timeout=3600,
)
def download_cpc() -> dict:
    """Download CPC independently to Modal and return a hash for cross-verification."""
    import hashlib
    import os

    import gdown

    target = DATA_ROOT / "datasets" / "cpc" / "CPCDataset.tar.gz"
    expected_sha256 = "dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tar.gz.download")
    if not target.is_file():
        temporary.unlink(missing_ok=True)
        result = gdown.download(
            id="1TMvuCSONEN1_9y7KnzKgy_7_fSFTHzyO",
            output=str(temporary),
            quiet=False,
        )
        if result is None:
            raise RuntimeError("CPC download failed")
        os.replace(temporary, target)
        volume.commit()
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"CPC archive hash mismatch: {digest}")
    result = {"path": str(target), "bytes": target.stat().st_size, "sha256": digest}
    print(result, flush=True)
    return result


@app.function(
    image=cpc_download_image,
    volumes={"/data": volume},
    cpu=2,
    memory=4096,
    timeout=1800,
)
def inspect_cpc_archive() -> dict:
    """Report the official archive layout before its training loader is used."""
    import json
    import shutil
    import tarfile

    source = DATA_ROOT / "datasets" / "cpc" / "CPCDataset.tar.gz"
    local = Path("/tmp/CPCDataset.tar.gz")
    shutil.copyfile(source, local)
    with tarfile.open(local, "r:gz") as archive:
        members = archive.getmembers()
        files = [member for member in members if member.isfile()]
        images = [
            member
            for member in files
            if Path(member.name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        text_files = [member for member in files if member.name.endswith(".txt")]
        annotations = [member for member in text_files if "CollectedAnnotationsRaw" in member.name]
        sample_text = archive.extractfile(annotations[0]).read().decode()
        sample = json.loads(sample_text)
        if isinstance(sample, str):
            sample = json.loads(sample)
    result = {
        "members": len(members),
        "files": len(files),
        "images": len(images),
        "annotations": len(annotations),
        "first_paths": [member.name for member in files[:10]],
        "sample_annotation": {
            "path": annotations[0].name,
            "score_shape": [len(sample["scores"]), len(sample["scores"][0])],
            "boxes": len(sample["bboxes"]),
            "first_box": sample["bboxes"][0],
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=teacher_image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=900,
)
def verify_teacher_runtime() -> dict:
    """Fail fast if CUDA or either uploaded teacher model cannot initialize."""
    _import_project()
    import onnxruntime as ort

    from focalnet.teacher import teacher_session

    models = DATA_ROOT / "teacher-models"
    sessions = [
        teacher_session(models / "u2net.onnx", provider="cuda"),
        teacher_session(models / "face_detection_yunet_2023mar.onnx", provider="cuda"),
    ]
    return {
        "available_providers": ort.get_available_providers(),
        "models": [session.get_inputs()[0].shape for session in sessions],
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=16,
    memory=32768,
    timeout=21600,
)
def download_images(dataset_name: str = "open-images-v7-100k", expected: int = 100_000) -> dict:
    """Recreate the exact normalized image set from its public Open Images IDs."""
    _import_project()
    import hashlib
    import json
    import os
    import tempfile

    import httpx

    from focalnet.openimages import _image_bytes, _normalise_jpeg

    if Path(dataset_name).name != dataset_name or expected <= 0:
        raise ValueError("dataset_name must be one path component and expected must be positive")
    dataset = DATA_ROOT / dataset_name
    manifest = dataset / "images.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError("Upload the Open Images manifest before downloading images")
    records = [json.loads(line) for line in manifest.open() if line.strip()]
    if len(records) != expected or len({record["image_id"] for record in records}) != expected:
        raise ValueError(f"Expected {expected:,} unique Open Images records")
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=True)
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    client = httpx.Client(
        headers={"User-Agent": "focalnet/0.1"},
        timeout=60,
        limits=limits,
        follow_redirects=True,
    )

    def fetch(record: dict) -> bool:
        target = dataset / record["image"]
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]:
            return False
        data, width, height = _normalise_jpeg(
            _image_bytes(record["image_id"], client=client),
            record["Rotation"],
            640,
            88,
        )
        if (width, height) != (record["width"], record["height"]):
            raise ValueError(f"dimension mismatch for {record['image_id']}")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"content mismatch for {record['image_id']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
        return True

    created = 0
    try:
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(fetch, record) for record in records]
            for completed, future in enumerate(as_completed(futures), 1):
                created += int(future.result())
                if completed % 5_000 == 0:
                    volume.commit()
                    print(f"Downloaded {completed:,}/{expected:,} images", flush=True)
    finally:
        client.close()
    volume.commit()
    return {"created": created, "total": len(records), "images": str(images)}


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=16,
    memory=32768,
    timeout=86400,
)
def download_planned_images(
    dataset_name: str = "open-images-v7-extension-400k", expected: int = 400_000
) -> dict:
    """Download a balanced candidate plan directly into the persistent Modal Volume."""
    _import_project()

    from focalnet.openimages import AcquireConfig, download_candidates

    if Path(dataset_name).name != dataset_name or expected <= 0:
        raise ValueError("dataset_name must be one path component and expected must be positive")
    dataset = DATA_ROOT / dataset_name
    if not (dataset / "candidates-with-metadata.jsonl").is_file():
        raise FileNotFoundError("Upload the planned Open Images candidate manifest first")
    result = download_candidates(
        dataset,
        AcquireConfig(
            count=expected,
            max_side=640,
            workers=128,
            storage_limit_gib=100,
            min_free_gib=1,
            seed=43,
            jpeg_quality=88,
            mix="expanded",
        ),
    )
    volume.commit()
    return result


@app.function(
    image=teacher_image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=86400,
)
def label_extension(
    dataset_name: str = "open-images-v7-extension-400k",
    output_name: str = "teacher-open-images-v7-extension-400k-fp32",
) -> dict:
    """Generate U²-Net + YuNet maps on CUDA while persisting every result to the Volume."""
    _import_project()

    from focalnet.data import label_images
    from focalnet.teacher import TeacherConfig

    if any(Path(name).name != name for name in (dataset_name, output_name)):
        raise ValueError("Dataset names must each be one path component")
    models = DATA_ROOT / "teacher-models"
    output = DATA_ROOT / output_name
    result = label_images(
        DATA_ROOT / dataset_name / "images",
        output,
        models / "u2net.onnx",
        models / "face_detection_yunet_2023mar.onnx",
        TeacherConfig(face_weight=3, score_threshold=0.85),
        threads=1,
        workers=8,
        provider="cuda",
        resume=(output / "provenance.json").is_file(),
    )
    volume.commit()
    return result


@app.function(
    image=teacher_image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=86400,
)
def label_extension_shard(
    start_index: int,
    end_index: int,
    dataset_name: str = "open-images-v7-extension-400k",
    output_prefix: str = "teacher-open-images-v7-extension-400k-fp32-shard",
    expected: int = 400_000,
) -> dict:
    """Label one disjoint sorted range so several L4s can work concurrently."""
    _import_project()

    from focalnet.data import label_images
    from focalnet.teacher import TeacherConfig

    if any(Path(name).name != name for name in (dataset_name, output_prefix)):
        raise ValueError("Dataset names must each be one path component")
    if not 0 <= start_index < end_index <= expected:
        raise ValueError("Shard range must satisfy 0 <= start < end <= expected")
    dataset = DATA_ROOT / dataset_name
    paths = _planned_image_paths(dataset, expected)[start_index:end_index]
    output_name = f"{output_prefix}-{start_index:06d}-{end_index:06d}"
    # The original serial run traversed the same sorted paths. Reuse its partial
    # first range instead of discarding tens of thousands of completed labels.
    if start_index == 0 and output_prefix == "teacher-open-images-v7-extension-400k-fp32-shard":
        output = DATA_ROOT / "teacher-open-images-v7-extension-400k-fp32"
    else:
        output = DATA_ROOT / output_name
    result = label_images(
        dataset / "images",
        output,
        DATA_ROOT / "teacher-models" / "u2net.onnx",
        DATA_ROOT / "teacher-models" / "face_detection_yunet_2023mar.onnx",
        TeacherConfig(face_weight=3, score_threshold=0.85),
        threads=1,
        workers=8,
        provider="cuda",
        resume=(output / "provenance.json").is_file(),
        source_paths=paths,
    )
    status = {
        **result,
        "dataset_name": dataset_name,
        "start_index": start_index,
        "end_index": end_index,
        "input_count": len(paths),
    }
    (output / "status.json").write_text(__import__("json").dumps(status, indent=2) + "\n")
    volume.commit()
    return status


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=8,
    memory=16384,
    timeout=21600,
)
def merge_extension_shards(
    dataset_name: str = "open-images-v7-extension-400k",
    shard_prefix: str = "teacher-open-images-v7-extension-400k-fp32-shard",
    output_name: str = "teacher-open-images-v7-extension-400k-fp32-parallel",
    expected: int = 400_000,
    shard_size: int = 100_000,
) -> dict:
    """Verify disjoint completed shards and create one deduplicated label manifest."""
    _import_project()
    import json
    import os
    import shutil

    from focalnet.data import read_manifest, sha256_file, write_manifest

    if any(Path(name).name != name for name in (dataset_name, shard_prefix, output_name)):
        raise ValueError("Dataset names must each be one path component")
    if expected <= 0 or shard_size <= 0 or expected % shard_size:
        raise ValueError("Expected count must be positive and divisible by shard size")
    dataset = DATA_ROOT / dataset_name
    all_paths = _planned_image_paths(dataset, expected)
    image_root = dataset / "images"
    expected_images = {str(path.relative_to(image_root)) for path in all_paths}
    merged_by_group = {}
    actual_images = set()
    duplicate_images = []
    outside_images = []
    source_metadata = []
    source_records = 0
    for start in range(0, expected, shard_size):
        end = start + shard_size
        if start == 0 and shard_prefix == "teacher-open-images-v7-extension-400k-fp32-shard":
            shard = DATA_ROOT / "teacher-open-images-v7-extension-400k-fp32"
        else:
            shard = DATA_ROOT / f"{shard_prefix}-{start:06d}-{end:06d}"
        status_path = shard / "status.json"
        labels_path = shard / "labels.jsonl"
        if not status_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"Incomplete labeling shard: {shard}")
        status = json.loads(status_path.read_text())
        required_status = {
            "dataset_name": dataset_name,
            "start_index": start,
            "end_index": end,
            "input_count": shard_size,
        }
        if any(status.get(key) != value for key, value in required_status.items()):
            raise ValueError(f"Shard status does not match its range: {status_path}")
        allowed = {str(path.relative_to(image_root)) for path in all_paths[start:end]}
        raw_records = [json.loads(line) for line in labels_path.open() if line.strip()]
        records = read_manifest(labels_path, validate_files=True)
        if status.get("total") != len(records) or len(raw_records) != len(records):
            raise ValueError(f"Shard status record count does not match: {status_path}")
        misplaced = 0
        for raw_record in raw_records:
            path = Path(os.path.abspath(shard / raw_record["image"]))
            if not path.is_relative_to(image_root):
                outside_images.append(str(path))
                continue
            relative = str(path.relative_to(image_root))
            if relative in actual_images:
                duplicate_images.append(relative)
            actual_images.add(relative)
            if relative not in allowed:
                misplaced += 1
        source_records += len(records)
        for record in records:
            merged_by_group.setdefault(record["group"], record)
        source_metadata.append(
            {
                "start_index": start,
                "end_index": end,
                "records": len(records),
                "records_outside_nominal_range": misplaced,
                "labels_sha256": sha256_file(labels_path),
            }
        )
    missing_images = expected_images - actual_images
    extra_images = actual_images - expected_images
    if outside_images or duplicate_images or missing_images or extra_images:
        examples = (
            outside_images[:1]
            or duplicate_images[:1]
            or sorted(missing_images)[:1]
            or sorted(extra_images)[:1]
        )
        raise ValueError(
            "Label shards do not cover the planned dataset exactly: "
            f"outside_root={len(outside_images):,}, duplicates={len(duplicate_images):,}, "
            f"missing={len(missing_images):,}, extra={len(extra_images):,}; "
            f"examples={examples}"
        )
    if source_records < expected * 0.99:
        raise ValueError(
            f"Shard manifests contain only {source_records:,} records for {expected:,} inputs"
        )
    merged = list(merged_by_group.values())
    output = DATA_ROOT / output_name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    write_manifest(output / "labels.jsonl", merged)
    metadata = {
        "format_version": 1,
        "target": "face-mass-mixture-v1",
        "merge": "parallel-sorted-ranges-v1",
        "dataset_name": dataset_name,
        "expected_inputs": expected,
        "source_records": source_records,
        "records": len(merged),
        "duplicate_groups_removed": source_records - len(merged),
        "shards": source_metadata,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    volume.commit()
    return metadata


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L40S",
    cpu=12,
    memory=32768,
    timeout=86400,
)
def train_model(run_name: str = "repvit-m0-9-v1", epochs: int = 20) -> dict:
    """Split, train, export FP32 ONNX, and create a calibrated INT8 model."""
    _import_project()
    import json
    import os
    import shutil

    from focalnet.data import split_manifest
    from focalnet.exporting import export_model, quantize_model
    from focalnet.model import ModelConfig
    from focalnet.training import TrainConfig, train

    volume_teacher = DATA_ROOT / "teacher-open-images-v7-100k-fp32"
    volume_labels = volume_teacher / "labels.jsonl"
    local_root = Path("/tmp/focalnet-data")
    local_teacher = local_root / "teacher-open-images-v7-100k-fp32"
    labels = local_teacher / "labels.jsonl"
    splits = local_root / "splits-v1"
    output = DATA_ROOT / "runs" / run_name
    if not volume_labels.is_file():
        raise FileNotFoundError("Upload the teacher labels before training")

    records = [json.loads(line) for line in volume_labels.open() if line.strip()]
    if len(records) != 100_000:
        raise ValueError(f"Expected 100,000 teacher records, found {len(records):,}")
    shutil.rmtree(local_root, ignore_errors=True)
    local_teacher.mkdir(parents=True)
    shutil.copyfile(volume_labels, labels)

    def stage(record: dict) -> None:
        for field in ("image", "importance"):
            source = Path(os.path.abspath(volume_teacher / record[field]))
            target = Path(os.path.abspath(local_teacher / record[field]))
            if not target.is_relative_to(local_root):
                raise ValueError(f"Training path escapes local staging directory: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(stage, record) for record in records]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 5_000 == 0:
                print(f"Staged {completed:,}/100,000 samples to local SSD", flush=True)

    split_manifest(labels, splits, validation_fraction=0.1, seed=42, validate_files=False)
    resume = output / "last.pt"
    if output.exists() and not resume.is_file():
        shutil.rmtree(output)

    result = train(
        splits / "train.jsonl",
        splits / "val.jsonl",
        output,
        ModelConfig("repvit_m0_9", 48),
        TrainConfig(
            epochs=epochs,
            batch_size=128,
            learning_rate=3e-4,
            weight_decay=1e-4,
            freeze_encoder_epochs=1,
            face_sampling_weight=2,
            workers=12,
            threads=12,
            device="cuda",
            pretrained=True,
            amp=True,
        ),
        resume if resume.is_file() else None,
        validate_files=False,
    )
    fp32 = output / "focalnet.onnx"
    int8 = output / "focalnet-int8.onnx"
    export_result = export_model(Path(result["checkpoint"]), fp32)
    quantize_result = quantize_model(fp32, splits / "train.jsonl", int8, samples=128, seed=42)
    volume.commit()
    return {"training": result, "export": export_result, "quantization": quantize_result}


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=8,
    memory=16384,
    timeout=3600,
)
def export_run(run_name: str = "repvit-m0-9-v1") -> dict:
    """Export an already-trained checkpoint without restaging or retraining the dataset."""
    _import_project()

    from focalnet.exporting import export_model

    output = DATA_ROOT / "runs" / run_name
    checkpoint = output / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")

    fp32 = output / "focalnet.onnx"
    for artifact in (fp32, fp32.with_suffix(".onnx.json")):
        artifact.unlink(missing_ok=True)
    export_result = export_model(checkpoint, fp32)
    volume.commit()
    return {"export": export_result}


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=8,
    memory=16384,
    timeout=3600,
)
def quantize_run(
    run_name: str = "repvit-m0-9-500k-v2",
    output_name: str = "focalnet-int8-u8.onnx",
    samples: int = 512,
    activation: str = "u8",
) -> dict:
    """Requantize an exported run without repeating training or FP32 export."""
    _import_project()
    import random
    import shutil

    from focalnet.data import read_manifest, write_manifest
    from focalnet.exporting import quantize_model

    if any(Path(name).name != name for name in (run_name, output_name)):
        raise ValueError("Run and output names must each be one component")
    teacher = DATA_ROOT / "teacher-open-images-v7-100k-fp32"
    output = DATA_ROOT / "runs" / run_name
    fp32 = output / "focalnet.onnx"
    if not fp32.is_file():
        raise FileNotFoundError(f"Missing FP32 model: {fp32}")

    calibration_root = Path("/tmp/focalnet-requantization")
    shutil.rmtree(calibration_root, ignore_errors=True)
    records = read_manifest(teacher / "labels.jsonl", validate_files=False)
    groups = sorted({record["group"] for record in records})
    random.Random(42).shuffle(groups)
    validation_count = max(1, min(len(groups) - 1, int(len(groups) * 0.1 + 0.5)))
    validation = set(groups[:validation_count])
    records = [record for record in records if record["group"] not in validation]
    random.Random(42).shuffle(records)
    calibration = calibration_root / "labels.jsonl"
    write_manifest(calibration, records[:samples])

    quantized = output / output_name
    quantized.unlink(missing_ok=True)
    quantized.with_suffix(quantized.suffix + ".json").unlink(missing_ok=True)
    result = quantize_model(
        fp32,
        calibration,
        quantized,
        samples=samples,
        seed=42,
        activation=activation,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=8,
    memory=32768,
    timeout=86400,
)
def pack_expanded_dataset(
    extension_name: str = "teacher-open-images-v7-extension-400k-fp32-parallel",
    dataset_name: str = "expanded-open-images-v2",
    shard_size: int = 5_000,
    part_index: int = 0,
    part_count: int = 1,
    pack_workers: int = 16,
) -> dict:
    """Build resumable tar shards for fast local-SSD staging of the expanded dataset."""
    _import_project()
    import json
    import os
    import random
    import shutil
    import tarfile
    import tempfile

    from focalnet.data import read_manifest, sha256_file, write_manifest

    if any(Path(name).name != name for name in (extension_name, dataset_name)):
        raise ValueError("Dataset names must each be one path component")
    if shard_size <= 0 or pack_workers <= 0:
        raise ValueError("shard_size and pack_workers must be positive")
    if part_count <= 0 or not 0 <= part_index < part_count:
        raise ValueError("Packing part must satisfy 0 <= part_index < part_count")
    old_labels = DATA_ROOT / "teacher-open-images-v7-100k-fp32" / "labels.jsonl"
    extension_labels = DATA_ROOT / extension_name / "labels.jsonl"
    old = read_manifest(old_labels, validate_files=False)
    extension = read_manifest(extension_labels, validate_files=False)
    if len(old) != 100_000:
        raise ValueError(f"Expected 100,000 original records, found {len(old):,}")

    groups = sorted({record["group"] for record in old})
    random.Random(42).shuffle(groups)
    validation_count = max(1, min(len(groups) - 1, int(len(groups) * 0.1 + 0.5)))
    validation_groups = set(groups[:validation_count])
    old_groups = set(groups)
    validation = [record for record in old if record["group"] in validation_groups]
    training = [record for record in old if record["group"] not in validation_groups]
    seen = set(old_groups)
    duplicate_groups = 0
    for record in extension:
        if record["group"] in seen:
            duplicate_groups += 1
            continue
        seen.add(record["group"])
        training.append(record)

    dataset = DATA_ROOT / "datasets" / dataset_name
    shards = dataset / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    expected_metadata = {
        "format_version": 1,
        "original_labels_sha256": sha256_file(old_labels),
        "extension_labels_sha256": sha256_file(extension_labels),
        "training": len(training),
        "validation": len(validation),
        "duplicate_extension_groups_removed": duplicate_groups,
        "shard_size": shard_size,
    }
    metadata_path = dataset / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != expected_metadata:
            raise ValueError(f"Existing packed dataset differs: {dataset}")
    else:
        metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n")

    def packed_records(records: list[dict]) -> list[dict]:
        packed = []
        for record in records:
            item = dict(record)
            item["image"] = str(dataset / "images" / f"{record['group']}.jpg")
            item["importance"] = str(dataset / "maps" / f"{record['group']}.npy")
            packed.append(item)
        return packed

    manifests = {
        "train": (training, packed_records(training)),
        "val": (validation, packed_records(validation)),
    }
    for split, (_, packed) in manifests.items():
        manifest = dataset / f"{split}.jsonl"
        if not manifest.exists():
            write_manifest(manifest, packed)

    total_shards = sum(
        (len(source) + shard_size - 1) // shard_size for source, _ in manifests.values()
    )
    jobs = []
    job_index = 0
    for split, (source, packed) in manifests.items():
        for start in range(0, len(source), shard_size):
            number = start // shard_size
            target = shards / f"{split}-{number:05d}.tar"
            selected = job_index % part_count == part_index
            job_index += 1
            if not selected or target.is_file():
                continue
            jobs.append(
                (
                    target,
                    source[start : start + shard_size],
                    packed[start : start + shard_size],
                )
            )

    def pack_shard(target: Path, source: list[dict], packed: list[dict]) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with tarfile.open(temporary_path, "w") as archive:
                for source_record, packed_record in zip(source, packed, strict=True):
                    archive.add(
                        source_record["image"],
                        arcname=Path(packed_record["image"]).relative_to(dataset),
                    )
                    archive.add(
                        source_record["importance"],
                        arcname=Path(packed_record["importance"]).relative_to(dataset),
                    )
            pending = target.with_suffix(".tar.tmp")
            shutil.copyfile(temporary_path, pending)
            os.replace(pending, target)
            return target
        finally:
            temporary_path.unlink(missing_ok=True)

    created = 0
    with ThreadPoolExecutor(max_workers=min(pack_workers, len(jobs) or 1)) as executor:
        futures = {
            executor.submit(pack_shard, target, source, packed): target
            for target, source, packed in jobs
        }
        for future in as_completed(futures):
            try:
                target = future.result()
                created += 1
                print(f"Packed {target.name} ({created:,} new, {total_shards:,} total)", flush=True)
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    volume.commit()
    return {
        **expected_metadata,
        "shards": total_shards,
        "created": created,
        "part_index": part_index,
        "part_count": part_count,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L40S",
    cpu=12,
    memory=32768,
    timeout=86400,
)
def train_expanded(
    dataset_name: str = "expanded-open-images-v2",
    run_name: str = "repvit-m0-9-500k-v2",
    epochs: int = 10,
) -> dict:
    """Fine-tune the 100k checkpoint on 490k training images and its fixed validation set."""
    _import_project()
    import json
    import shutil
    import tarfile

    from focalnet.exporting import export_model
    from focalnet.model import ModelConfig
    from focalnet.training import TrainConfig, train

    if any(Path(name).name != name for name in (dataset_name, run_name)) or epochs <= 0:
        raise ValueError("Names must each be one component and epochs must be positive")
    packed = DATA_ROOT / "datasets" / dataset_name
    local = Path("/tmp/focalnet-expanded")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    for manifest in ("train.jsonl", "val.jsonl"):
        with (packed / manifest).open() as source, (local / manifest).open("w") as target:
            for line in source:
                record = json.loads(line)
                group = record["group"]
                record["image"] = str(local / "images" / f"{group}.jpg")
                record["importance"] = str(local / "maps" / f"{group}.npy")
                target.write(json.dumps(record, separators=(",", ":")) + "\n")
    metadata = json.loads((packed / "metadata.json").read_text())
    shard_size = int(metadata["shard_size"])
    expected_shards = {
        f"{split}-{number:05d}.tar"
        for split, count in (
            ("train", int(metadata["training"])),
            ("val", int(metadata["validation"])),
        )
        for number in range((count + shard_size - 1) // shard_size)
    }
    shards = sorted((packed / "shards").glob("*.tar"))
    actual_shards = {shard.name for shard in shards}
    if actual_shards != expected_shards:
        missing = sorted(expected_shards - actual_shards)
        unexpected = sorted(actual_shards - expected_shards)
        raise FileNotFoundError(
            f"Packed dataset is incomplete: expected {len(expected_shards):,} shards, "
            f"found {len(actual_shards):,}; missing={missing}, unexpected={unexpected}"
        )
    for index, shard in enumerate(shards, 1):
        temporary = local / "current.tar"
        shutil.copyfile(shard, temporary)
        with tarfile.open(temporary) as archive:
            archive.extractall(local, filter="data")
        temporary.unlink()
        print(f"Staged shard {index:,}/{len(shards):,} to local SSD", flush=True)

    output = DATA_ROOT / "runs" / run_name
    resume = output / "last.pt"
    if output.exists() and not resume.is_file():
        shutil.rmtree(output)
    initializer = DATA_ROOT / "runs" / "repvit-m0-9-v1" / "best.pt"
    result = train(
        local / "train.jsonl",
        local / "val.jsonl",
        output,
        ModelConfig("repvit_m0_9", 48),
        TrainConfig(
            epochs=epochs,
            batch_size=128,
            learning_rate=1e-4,
            weight_decay=1e-4,
            freeze_encoder_epochs=0,
            face_sampling_weight=2,
            workers=12,
            threads=12,
            device="cuda",
            pretrained=False,
            amp=True,
        ),
        resume=resume if resume.is_file() else None,
        initialize=None if resume.is_file() else initializer,
        validate_files=False,
    )
    fp32 = output / "focalnet.onnx"
    for artifact in (fp32, fp32.with_suffix(".onnx.json")):
        artifact.unlink(missing_ok=True)
    export_result = export_model(Path(result["checkpoint"]), fp32)
    volume.commit()
    return {"training": result, "export": export_result}


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=12,
    memory=32768,
    timeout=21600,
)
def train_human_crops(
    run_name: str = "repvit-m0-9-500k-gaic-v1",
    epochs: int = 30,
) -> dict:
    """Train and evaluate a frozen-base human crop ranker on GAICD v2."""
    _import_project()
    import json
    import shutil
    import zipfile

    import torch

    from focalnet.data import sha256_file
    from focalnet.human_exporting import export_human_model
    from focalnet.human_training import (
        GAICDataset,
        HumanTrainConfig,
        evaluate_human_model,
        train_human_ranker,
    )
    from focalnet.ranking import load_human_checkpoint

    if Path(run_name).name != run_name or epochs <= 0:
        raise ValueError("Run name must be one component and epochs must be positive")
    volume_archive = DATA_ROOT / "datasets" / "gaic-v2" / "GAIC.zip"
    if not volume_archive.is_file():
        raise FileNotFoundError(f"Missing GAICD archive: {volume_archive}")
    expected_sha256 = "b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b"
    if sha256_file(volume_archive) != expected_sha256:
        raise ValueError("GAICD archive hash differs from the reviewed journal-version download")
    local = Path("/tmp/focalnet-gaic-v2")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir()
    archive = local / "GAIC.zip"
    shutil.copyfile(volume_archive, archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in GAICD archive: {member.filename}")
        source.extractall(local)
    dataset = local / "GAIC"
    expected_images = {"train": 2636, "val": 200, "test": 500}
    actual_images = {
        split: len(list((dataset / "images" / split).glob("*.jpg"))) for split in expected_images
    }
    if actual_images != expected_images:
        raise ValueError(f"Unexpected GAICD split counts: {actual_images}")

    output = DATA_ROOT / "runs" / run_name
    resume = output / "last.pt"
    if output.exists() and not resume.is_file():
        shutil.rmtree(output)
    base = DATA_ROOT / "runs" / "repvit-m0-9-500k-v2" / "best.pt"
    result = train_human_ranker(
        dataset,
        base,
        output,
        HumanTrainConfig(
            epochs=epochs,
            batch_size=64,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=42,
            workers=8,
            threads=12,
            device="cuda",
            amp=True,
        ),
        resume=resume if resume.is_file() else None,
    )
    volume.commit()
    model, _ = load_human_checkpoint(result["checkpoint"])
    model.to("cuda")
    evaluation = evaluate_human_model(
        model,
        GAICDataset(dataset, "test"),
        batch_size=64,
        workers=8,
        device=torch.device("cuda"),
    )
    evaluation.update(
        {
            "dataset": "GAICD journal version",
            "split": "test",
            "archive_sha256": expected_sha256,
            "images_expected": expected_images,
        }
    )
    (output / "evaluation-gaicd-test.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    artifact = output / "focalnet-human.onnx"
    artifact.unlink(missing_ok=True)
    artifact.with_suffix(artifact.suffix + ".json").unlink(missing_ok=True)
    export_result = export_human_model(Path(result["checkpoint"]), artifact)
    volume.commit()
    return {"training": result, "evaluation": evaluation, "export": export_result}


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=12,
    memory=32768,
    timeout=3600,
)
def validate_cpc_dataset() -> dict:
    """Extract CPC and validate every paired image, annotation, crop, and split."""
    _import_project()
    import hashlib
    import json
    import shutil
    import tarfile

    import numpy as np

    from focalnet.cpc import CPCDataset, cpc_fingerprint
    from focalnet.data import sha256_file

    expected_sha256 = "dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281"
    archive = DATA_ROOT / "datasets" / "cpc" / "CPCDataset.tar.gz"
    if sha256_file(archive) != expected_sha256:
        raise ValueError("CPC archive hash differs from the reviewed official download")
    local = Path("/tmp/focalnet-cpc-validation")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe path in CPC archive: {member.name}")
        source.extractall(local, filter="data")
    root = local / "CPCDataset"
    train = CPCDataset(root, "train", seed=42)
    validation = CPCDataset(root, "val", seed=42)
    if (len(train), len(validation)) != (9717, 1080):
        raise ValueError(f"Unexpected CPC splits: {len(train)} train, {len(validation)} val")
    scores = np.concatenate(
        [record[2] for dataset in (train, validation) for record in dataset.records]
    )
    file_hashes = {}
    for dataset in (train, validation):
        for path, _, _, _ in dataset.records:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            file_hashes.setdefault(digest, []).append(path.name)
    duplicates = {digest: names for digest, names in file_hashes.items() if len(names) > 1}
    result = {
        "archive_sha256": expected_sha256,
        "train_images": len(train),
        "validation_images": len(validation),
        "rated_crops": int(len(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "exact_duplicate_groups": len(duplicates),
        "dataset_fingerprint": cpc_fingerprint(root, seed=42),
    }
    report = DATA_ROOT / "datasets" / "cpc" / "validation.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=12,
    memory=32768,
    timeout=10800,
)
def pretrain_cpc_ranker(
    run_name: str = "repvit-m0-9-500k-cpc-v1",
    epochs: int = 20,
) -> dict:
    """Pretrain the crop ranker on CPC without accessing any GAICD split."""
    _import_project()
    import shutil
    import tarfile

    from focalnet.cpc import CPCDataset, train_cpc_ranker
    from focalnet.data import sha256_file
    from focalnet.human_training import HumanTrainConfig

    if Path(run_name).name != run_name or epochs <= 0:
        raise ValueError("Run name must be one component and epochs must be positive")
    expected_sha256 = "dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281"
    archive = DATA_ROOT / "datasets" / "cpc" / "CPCDataset.tar.gz"
    if sha256_file(archive) != expected_sha256:
        raise ValueError("CPC archive hash differs from the reviewed official download")
    local = Path("/tmp/focalnet-cpc-pretrain")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe path in CPC archive: {member.name}")
        source.extractall(local, filter="data")
    root = local / "CPCDataset"
    if (len(CPCDataset(root, "train", seed=42)), len(CPCDataset(root, "val", seed=42))) != (
        9717,
        1080,
    ):
        raise ValueError("Unexpected CPC split counts")
    output = DATA_ROOT / "runs" / run_name
    resume = output / "last.pt"
    if output.exists() and not resume.is_file():
        shutil.rmtree(output)
    result = train_cpc_ranker(
        root,
        DATA_ROOT / "runs" / "repvit-m0-9-500k-v2" / "best.pt",
        output,
        HumanTrainConfig(
            epochs=epochs,
            batch_size=64,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=42,
            workers=8,
            threads=12,
            device="cuda",
            amp=True,
        ),
        resume=resume if resume.is_file() else None,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=12,
    memory=32768,
    timeout=21600,
)
def train_cpc_then_gaic(
    cpc_run_name: str = "repvit-m0-9-500k-cpc-v1",
    final_run_name: str = "repvit-m0-9-500k-cpc-gaic-v2",
    cpc_epochs: int = 20,
    gaic_epochs: int = 30,
) -> dict:
    """Pretrain on CPC, fine-tune on GAICD, then evaluate both held-out splits."""
    _import_project()
    import json
    import shutil
    import tarfile
    import zipfile

    import torch

    from focalnet.cpc import CPCDataset, train_cpc_ranker
    from focalnet.data import sha256_file
    from focalnet.human_exporting import export_human_model
    from focalnet.human_training import (
        GAICDataset,
        HumanTrainConfig,
        evaluate_human_model,
        train_human_ranker,
    )
    from focalnet.ranking import load_human_checkpoint

    if any(Path(name).name != name for name in (cpc_run_name, final_run_name)):
        raise ValueError("Run names must each be one path component")
    if min(cpc_epochs, gaic_epochs) <= 0:
        raise ValueError("Training epochs must be positive")
    cpc_sha256 = "dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281"
    gaic_sha256 = "b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b"
    cpc_archive = DATA_ROOT / "datasets" / "cpc" / "CPCDataset.tar.gz"
    gaic_archive = DATA_ROOT / "datasets" / "gaic-v2" / "GAIC.zip"
    if sha256_file(cpc_archive) != cpc_sha256 or sha256_file(gaic_archive) != gaic_sha256:
        raise ValueError("CPC or GAICD archive hash differs from its reviewed download")

    local = Path("/tmp/focalnet-cpc-gaic")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir()
    with tarfile.open(cpc_archive, "r:gz") as source:
        for member in source.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe path in CPC archive: {member.name}")
        source.extractall(local, filter="data")
    with zipfile.ZipFile(gaic_archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in GAICD archive: {member.filename}")
        source.extractall(local)
    cpc_root, gaic_root = local / "CPCDataset", local / "GAIC"
    cpc_train = CPCDataset(cpc_root, "train", seed=42)
    cpc_val = CPCDataset(cpc_root, "val", seed=42)
    if (len(cpc_train), len(cpc_val)) != (9717, 1080):
        raise ValueError(f"Unexpected CPC splits: {len(cpc_train)} train, {len(cpc_val)} val")
    gaic_counts = {
        split: len(list((gaic_root / "images" / split).glob("*.jpg")))
        for split in ("train", "val", "test")
    }
    if gaic_counts != {"train": 2636, "val": 200, "test": 500}:
        raise ValueError(f"Unexpected GAICD splits: {gaic_counts}")
    protected_hashes = {
        sha256_file(path)
        for split in ("val", "test")
        for path in (gaic_root / "images" / split).glob("*.jpg")
    }
    cpc_hashes = {
        sha256_file(path)
        for path in (cpc_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    overlap = protected_hashes & cpc_hashes
    if overlap:
        raise ValueError(f"CPC overlaps {len(overlap)} GAICD validation/test images")

    base = DATA_ROOT / "runs" / "repvit-m0-9-500k-v2" / "best.pt"
    cpc_output = DATA_ROOT / "runs" / cpc_run_name
    cpc_resume = cpc_output / "last.pt"
    if cpc_output.exists() and not cpc_resume.is_file():
        shutil.rmtree(cpc_output)
    cpc_result = train_cpc_ranker(
        cpc_root,
        base,
        cpc_output,
        HumanTrainConfig(
            epochs=cpc_epochs,
            batch_size=64,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=42,
            workers=8,
            threads=12,
            device="cuda",
            amp=True,
        ),
        resume=cpc_resume if cpc_resume.is_file() else None,
    )
    volume.commit()

    final_output = DATA_ROOT / "runs" / final_run_name
    final_resume = final_output / "last.pt"
    if final_output.exists() and not final_resume.is_file():
        shutil.rmtree(final_output)
    final_result = train_human_ranker(
        gaic_root,
        None,
        final_output,
        HumanTrainConfig(
            epochs=gaic_epochs,
            batch_size=64,
            learning_rate=1e-4,
            weight_decay=1e-4,
            seed=42,
            workers=8,
            threads=12,
            device="cuda",
            amp=True,
        ),
        resume=final_resume if final_resume.is_file() else None,
        initialize=None if final_resume.is_file() else Path(cpc_result["checkpoint"]),
    )
    volume.commit()

    cpc_model, _ = load_human_checkpoint(cpc_result["checkpoint"])
    cpc_model.to("cuda")
    cpc_evaluation = evaluate_human_model(
        cpc_model, cpc_val, batch_size=64, workers=8, device=torch.device("cuda")
    )
    final_model, _ = load_human_checkpoint(final_result["checkpoint"])
    final_model.to("cuda")
    final_cpc_evaluation = evaluate_human_model(
        final_model, cpc_val, batch_size=64, workers=8, device=torch.device("cuda")
    )
    gaic_evaluation = evaluate_human_model(
        final_model,
        GAICDataset(gaic_root, "test"),
        batch_size=64,
        workers=8,
        device=torch.device("cuda"),
    )
    evaluation = {
        "cpc_archive_sha256": cpc_sha256,
        "gaic_archive_sha256": gaic_sha256,
        "cpc_split": {"train": len(cpc_train), "validation": len(cpc_val)},
        "exact_overlap_with_gaicd_validation_test": 0,
        "cpc_pretrain_validation": cpc_evaluation,
        "final_cpc_validation": final_cpc_evaluation,
        "final_gaicd_test": gaic_evaluation,
    }
    (final_output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    artifact = final_output / "focalnet-human.onnx"
    artifact.unlink(missing_ok=True)
    artifact.with_suffix(artifact.suffix + ".json").unlink(missing_ok=True)
    export_result = export_human_model(Path(final_result["checkpoint"]), artifact)
    volume.commit()
    return {
        "cpc_training": cpc_result,
        "gaic_finetuning": final_result,
        "evaluation": evaluation,
        "export": export_result,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=12,
    memory=32768,
    timeout=10800,
)
def finetune_cpc_on_gaic_candidate(
    run_name: str = "repvit-m0-9-500k-cpc-gaic-lr3e4-candidate",
    epochs: int = 30,
) -> dict:
    """Tune CPC transfer on GAICD validation without reading the GAICD test split."""
    _import_project()
    import shutil
    import zipfile

    from focalnet.data import sha256_file
    from focalnet.human_training import HumanTrainConfig, train_human_ranker

    if Path(run_name).name != run_name or epochs <= 0:
        raise ValueError("Run name must be one component and epochs must be positive")
    gaic_sha256 = "b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b"
    archive = DATA_ROOT / "datasets" / "gaic-v2" / "GAIC.zip"
    if sha256_file(archive) != gaic_sha256:
        raise ValueError("GAICD archive hash differs from the reviewed download")
    local = Path("/tmp/focalnet-cpc-gaic-candidate")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in GAICD archive: {member.filename}")
        source.extractall(local)
    output = DATA_ROOT / "runs" / run_name
    resume = output / "last.pt"
    if output.exists() and not resume.is_file():
        shutil.rmtree(output)
    result = train_human_ranker(
        local / "GAIC",
        None,
        output,
        HumanTrainConfig(
            epochs=epochs,
            batch_size=64,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=42,
            workers=8,
            threads=12,
            device="cuda",
            amp=True,
        ),
        resume=resume if resume.is_file() else None,
        initialize=(
            None if resume.is_file() else DATA_ROOT / "runs" / "repvit-m0-9-500k-cpc-v1" / "best.pt"
        ),
    )
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4",
    cpu=12,
    memory=32768,
    timeout=3600,
)
def evaluate_cpc_gaic_candidate(
    run_name: str = "repvit-m0-9-500k-cpc-gaic-lr3e4-candidate",
) -> dict:
    """Evaluate a selected CPC transfer candidate once and export its best checkpoint."""
    _import_project()
    import json
    import shutil
    import tarfile
    import zipfile

    import torch

    from focalnet.cpc import CPCDataset
    from focalnet.data import sha256_file
    from focalnet.human_exporting import export_human_model
    from focalnet.human_training import GAICDataset, evaluate_human_model
    from focalnet.ranking import load_human_checkpoint

    if Path(run_name).name != run_name:
        raise ValueError("Run name must be one component")
    cpc_sha256 = "dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281"
    gaic_sha256 = "b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b"
    cpc_archive = DATA_ROOT / "datasets" / "cpc" / "CPCDataset.tar.gz"
    gaic_archive = DATA_ROOT / "datasets" / "gaic-v2" / "GAIC.zip"
    if sha256_file(cpc_archive) != cpc_sha256 or sha256_file(gaic_archive) != gaic_sha256:
        raise ValueError("CPC or GAICD archive hash differs from its reviewed download")

    output = DATA_ROOT / "runs" / run_name
    checkpoint = output / "best.pt"
    history_path = output / "history.json"
    if not checkpoint.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Candidate training is incomplete: {output}")
    history = json.loads(history_path.read_text())
    if not history:
        raise ValueError("Candidate history is empty")
    selected = max(history, key=lambda record: record["val_srcc"])

    local = Path("/tmp/focalnet-cpc-gaic-evaluation")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir()
    with tarfile.open(cpc_archive, "r:gz") as source:
        for member in source.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe path in CPC archive: {member.name}")
        source.extractall(local, filter="data")
    with zipfile.ZipFile(gaic_archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in GAICD archive: {member.filename}")
        source.extractall(local)

    cpc_validation = CPCDataset(local / "CPCDataset", "val", seed=42)
    gaic_test = GAICDataset(local / "GAIC", "test")
    if len(cpc_validation) != 1080 or len(gaic_test) != 500:
        raise ValueError(
            f"Unexpected evaluation split counts: CPC={len(cpc_validation)}, GAICD={len(gaic_test)}"
        )
    model, _ = load_human_checkpoint(checkpoint)
    model.to("cuda")
    device = torch.device("cuda")
    gaic_evaluation = evaluate_human_model(
        model, gaic_test, batch_size=64, workers=8, device=device
    )
    cpc_evaluation = evaluate_human_model(
        model, cpc_validation, batch_size=64, workers=8, device=device
    )
    evaluation = {
        "candidate": run_name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_validation_epoch": selected["epoch"],
        "selected_gaicd_validation": {
            key: value for key, value in selected.items() if key.startswith("val_")
        },
        "gaicd_test": gaic_evaluation,
        "cpc_validation_after_gaicd": cpc_evaluation,
        "cpc_archive_sha256": cpc_sha256,
        "gaic_archive_sha256": gaic_sha256,
    }
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    artifact = output / "focalnet-human.onnx"
    artifact.unlink(missing_ok=True)
    artifact.with_suffix(artifact.suffix + ".json").unlink(missing_ok=True)
    export_result = export_human_model(checkpoint, artifact)
    volume.commit()
    result = {"evaluation": evaluation, "export": export_result}
    print(json.dumps(result, indent=2), flush=True)
    return result
