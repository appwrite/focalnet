#!/usr/bin/env python3
"""Upload teacher maps to Modal in small, resumable transactions."""

import argparse
import hashlib
import json
from pathlib import Path

import modal


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def upload(
    volume_name: str,
    batch_size: int,
    teacher_path: Path,
    images_path: Path,
    state_path: Path,
    expected: int,
) -> None:
    root = Path(__file__).resolve().parent.parent
    downloads = (root / "downloads").resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    teacher = resolve(teacher_path)
    open_images = resolve(images_path)
    state_path = resolve(state_path)
    for path in (teacher, open_images):
        if not path.is_relative_to(downloads):
            raise ValueError(f"Dataset must be stored below {downloads}: {path}")
    fixed = [
        teacher / "labels.jsonl",
        teacher / "provenance.json",
        open_images / "images.jsonl",
        open_images / "provenance.json",
    ]
    maps = sorted((teacher / "maps").glob("*.npy"))
    for path in fixed:
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = sum(1 for line in (teacher / "labels.jsonl").open() if line.strip())
    if len(maps) != labels or (expected and labels != expected):
        wanted = f"{expected:,}" if expected else "the label count"
        raise ValueError(f"Expected {wanted} labels/maps, found {labels:,}/{len(maps):,}")
    files = fixed + maps
    total_bytes = sum(path.stat().st_size for path in files)
    fingerprint = {
        "volume": volume_name,
        "labels_sha256": sha256_file(teacher / "labels.jsonl"),
        "files": len(files),
        "bytes": total_bytes,
    }
    uploaded = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError(f"Upload state does not match this dataset: {state_path}")
        uploaded = int(state["uploaded"])

    volume = modal.Volume.from_name(volume_name)
    uploaded_bytes = sum(path.stat().st_size for path in files[:uploaded])
    for start in range(uploaded, len(files), batch_size):
        batch = files[start : start + batch_size]
        with volume.batch_upload(force=True) as transaction:
            for path in batch:
                remote = "/" + path.relative_to(downloads).as_posix()
                transaction.put_file(path, remote)
        uploaded = start + len(batch)
        uploaded_bytes += sum(path.stat().st_size for path in batch)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"fingerprint": fingerprint, "uploaded": uploaded}, indent=2) + "\n"
        )
        temporary.replace(state_path)
        print(
            f"Uploaded {uploaded:,}/{len(files):,} files "
            f"({uploaded_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB, "
            f"{uploaded / len(files):.1%})",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", default="focalnet-data")
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument(
        "--teacher", type=Path, default=Path("downloads/teacher-open-images-v7-100k-fp32")
    )
    parser.add_argument("--images", type=Path, default=Path("downloads/open-images-v7-100k"))
    parser.add_argument("--state", type=Path, default=Path("downloads/modal-upload-state.json"))
    parser.add_argument("--expected", type=int, default=100_000, help="Use 0 to infer from labels")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.expected < 0:
        parser.error("--batch-size must be positive and --expected nonnegative")
    upload(args.volume, args.batch_size, args.teacher, args.images, args.state, args.expected)


if __name__ == "__main__":
    main()
