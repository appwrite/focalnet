import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "publish-huggingface.py"


def load_publisher():
    spec = importlib.util.spec_from_file_location("publish_huggingface", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_card_declares_hub_metadata_and_reference_hashes():
    publisher = load_publisher()
    card = (REPO / "docs" / "huggingface-model-card.md").read_text()
    assert card.startswith("---\n")
    for field in (
        "license: mit",
        "library_name: onnxruntime",
        "pipeline_tag: image-to-image",
        "base_model: timm/repvit_m0_9",
    ):
        assert field in card
    for digest in publisher.EXPECTED_SHA256.values():
        assert digest in card
    assert "Microsoft FocalNet" not in card
    images = publisher.image_files(REPO / "docs" / "hub-images")
    assert images
    for picture in images:
        assert f"images/{picture.name}" in card


def test_reference_hashes_match_docs_and_fetch_script():
    publisher = load_publisher()
    docs = (REPO / "docs" / "results-500k.md").read_text() + (
        REPO / "docs" / "results-cpc-gaic-v2.md"
    ).read_text()
    fetch = (REPO / "web" / "scripts" / "fetch-model.ts").read_text()
    assert publisher.EXPECTED_SHA256["focalnet.onnx"] in docs
    assert publisher.EXPECTED_SHA256["focalnet.pt"] in docs
    assert publisher.EXPECTED_SHA256["focalnet-human.onnx"] in docs
    assert publisher.EXPECTED_SHA256["focalnet-human.pt"] in docs
    assert publisher.EXPECTED_SHA256["focalnet-human.onnx"] in fetch
    assert "https://huggingface.co/appwrite/focalnet/resolve/main/focalnet-human.onnx" in fetch


def test_prepare_staging_renames_checkpoints_and_rejects_hash_mismatch(tmp_path):
    publisher = load_publisher()
    importance = tmp_path / "importance"
    human = tmp_path / "human"
    importance.mkdir()
    human.mkdir()
    files = {
        importance / "focalnet.onnx": b"onnx-v1",
        importance / "focalnet.onnx.json": b'{"format_version": 1}',
        importance / "best.pt": b"importance-pt",
        human / "focalnet-human.onnx": b"onnx-v2",
        human / "focalnet-human.onnx.json": b'{"format_version": 2}',
        human / "best.pt": b"human-pt",
    }
    for path, payload in files.items():
        path.write_bytes(payload)
    expected = {
        "focalnet.onnx": publisher.sha256_file(importance / "focalnet.onnx"),
        "focalnet.pt": publisher.sha256_file(importance / "best.pt"),
        "focalnet-human.onnx": publisher.sha256_file(human / "focalnet-human.onnx"),
        "focalnet-human.pt": publisher.sha256_file(human / "best.pt"),
    }
    pictures = tmp_path / "images"
    pictures.mkdir()
    (pictures / "example.jpg").write_bytes(b"jpeg-bytes")
    card = tmp_path / "card.md"
    card.write_text("\n".join([*expected.values(), "images/example.jpg"]))
    staging = tmp_path / "hub"
    staged = publisher.prepare_staging(
        importance, human, card, staging, expected=expected, images_dir=pictures
    )
    assert (staging / "focalnet.pt").read_bytes() == b"importance-pt"
    assert (staging / "focalnet-human.pt").read_bytes() == b"human-pt"
    assert (staging / "images" / "example.jpg").read_bytes() == b"jpeg-bytes"
    assert staged["focalnet.onnx"] == expected["focalnet.onnx"]
    assert staged["focalnet-human.pt"] == expected["focalnet-human.pt"]
    assert staged["images/example.jpg"] == publisher.sha256_file(pictures / "example.jpg")

    (human / "focalnet-human.onnx").write_bytes(b"tampered")
    try:
        publisher.prepare_staging(
            importance, human, card, staging, expected=expected, images_dir=pictures
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("tampered ONNX should fail hash verification")
