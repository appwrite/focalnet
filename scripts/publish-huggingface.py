"""Stage and upload verified FocalNet checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

REPO_ID = "appwrite/focalnet"
DEFAULT_CARD = Path("docs/huggingface-model-card.md")
DEFAULT_IMAGES = Path("docs/hub-images")
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
EXPECTED_SHA256 = {
    "focalnet.onnx": "87becceb269a2973c359df789783be49a9f840f47170a015d3776d7c4145a2ce",
    "focalnet.pt": "d1942f0652f8ea85f75ffc0cb1bf40d7e70b38e7ad102ab0e6df5c2e07ce52cf",
    "focalnet-human.onnx": "59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439",
    "focalnet-human.pt": "82c4d695310c220e5f3bcfbf5e64d315a5e826cfd77e945e61e032460014480a",
}
HUB_WEIGHTS = (
    "focalnet.onnx",
    "focalnet.onnx.json",
    "focalnet.pt",
    "focalnet-human.onnx",
    "focalnet-human.onnx.json",
    "focalnet-human.pt",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path} hash {actual} does not match {expected}")
    return actual


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def image_files(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def staged_paths(staging: Path) -> list[str]:
    return sorted(str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file())


def prepare_staging(
    importance_dir: Path,
    human_dir: Path,
    card: Path,
    staging: Path,
    expected: dict[str, str] | None = None,
    images_dir: Path | None = None,
) -> dict[str, str]:
    expected = EXPECTED_SHA256 if expected is None else expected
    images_dir = DEFAULT_IMAGES if images_dir is None else images_dir
    if not card.is_file():
        raise FileNotFoundError(f"Missing model card: {card}")
    card_text = card.read_text()
    for digest in expected.values():
        if digest not in card_text:
            raise ValueError(f"Model card is missing expected hash {digest}")
    pictures = image_files(images_dir)
    for picture in pictures:
        reference = f"images/{picture.name}"
        if reference not in card_text:
            raise ValueError(f"Model card is missing image {reference}")

    sources = {
        "focalnet.onnx": importance_dir / "focalnet.onnx",
        "focalnet.onnx.json": importance_dir / "focalnet.onnx.json",
        "focalnet.pt": importance_dir / "best.pt",
        "focalnet-human.onnx": human_dir / "focalnet-human.onnx",
        "focalnet-human.onnx.json": human_dir / "focalnet-human.onnx.json",
        "focalnet-human.pt": human_dir / "best.pt",
    }
    for name, digest in expected.items():
        require_hash(sources[name], digest)

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    copy_file(card, staging / "README.md")
    for name, source in sources.items():
        copy_file(source, staging / name)
        if name in expected:
            require_hash(staging / name, expected[name])
    for picture in pictures:
        copy_file(picture, staging / "images" / picture.name)

    allowed = {"README.md", *HUB_WEIGHTS, *(f"images/{picture.name}" for picture in pictures)}
    missing = [name for name in allowed if not (staging / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Staging is missing {missing}")
    extra = [name for name in staged_paths(staging) if name not in allowed]
    if extra:
        raise ValueError(f"Unexpected staged files: {extra}")
    return {name: sha256_file(staging / name) for name in allowed}


def upload_staging(staging: Path, repo_id: str = REPO_ID, *, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    missing = [name for name in ("README.md", *HUB_WEIGHTS) if not (staging / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Staging is missing {missing}")
    for name, digest in EXPECTED_SHA256.items():
        require_hash(staging / name, digest)
    api = HfApi(token=token)
    commit = api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish FocalNet checkpoints, model card, and examples",
        allow_patterns=["README.md", "images/*", *HUB_WEIGHTS],
    )
    return getattr(commit, "oid", str(commit))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Publish verified FocalNet weights to Hugging Face")
    root.add_argument("--importance-dir", type=Path, required=True)
    root.add_argument("--human-dir", type=Path, required=True)
    root.add_argument("--card", type=Path, default=DEFAULT_CARD)
    root.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    root.add_argument("--staging", type=Path, required=True)
    root.add_argument("--repo-id", default=REPO_ID)
    root.add_argument("--upload", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    staged = prepare_staging(
        args.importance_dir,
        args.human_dir,
        args.card,
        args.staging,
        images_dir=args.images_dir,
    )
    for name, digest in staged.items():
        print(f"{digest}  {name}")
    if not args.upload:
        return
    commit = upload_staging(args.staging, args.repo_id)
    print(f"Uploaded to https://huggingface.co/{args.repo_id} ({commit})")


if __name__ == "__main__":
    main()
