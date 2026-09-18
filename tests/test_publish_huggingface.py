import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "publish-huggingface.py"
RENDER = REPO / "scripts" / "render-hub-examples.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_publisher():
    return load_script(SCRIPT, "publish_huggingface")


def write_release_inputs(tmp_path: Path) -> dict:
    publisher = load_publisher()
    importance = tmp_path / "importance"
    human = tmp_path / "human"
    pictures = tmp_path / "images"
    importance.mkdir()
    human.mkdir()
    pictures.mkdir()
    files = {
        importance / "focalnet.onnx": b"onnx-v1",
        importance / "best.pt": b"importance-pt",
        human / "focalnet-human.onnx": b"onnx-v2",
        human / "best.pt": b"human-pt",
    }
    for path, payload in files.items():
        path.write_bytes(payload)
    embedded = {
        "focalnet.onnx": {"format_version": 1, "task": "importance"},
        "focalnet-human.onnx": {"format_version": 2, "task": "human-crop-ranking"},
    }
    (importance / "focalnet.onnx.json").write_text(json.dumps(embedded["focalnet.onnx"]))
    (human / "focalnet-human.onnx.json").write_text(json.dumps(embedded["focalnet-human.onnx"]))
    (pictures / "example.jpg").write_bytes(b"jpeg-bytes")
    expected = {
        "focalnet.onnx": publisher.sha256_file(importance / "focalnet.onnx"),
        "focalnet.pt": publisher.sha256_file(importance / "best.pt"),
        "focalnet-human.onnx": publisher.sha256_file(human / "focalnet-human.onnx"),
        "focalnet-human.pt": publisher.sha256_file(human / "best.pt"),
    }
    card = tmp_path / "card.md"
    card.write_text(
        "\n".join(
            [
                *(f"`{name}` `{digest}`" for name, digest in expected.items()),
                "![example](images/example.jpg)",
            ]
        )
    )
    return {
        "publisher": publisher,
        "importance": importance,
        "human": human,
        "pictures": pictures,
        "card": card,
        "expected": expected,
        "embedded": embedded,
        "read_metadata": lambda path: json.loads(json.dumps(embedded[path.name])),
    }


def test_committed_card_stays_aligned_with_publisher_and_images():
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
    assert "Microsoft FocalNet" not in card
    publisher.require_paired_hashes(card, publisher.EXPECTED_SHA256)
    publisher.require_card_images(card, publisher.image_files(REPO / "docs" / "hub-images"))


def test_docs_and_fetch_script_keep_the_publisher_hashes():
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
    assert "huggingface.co/appwrite/focalnet" in fetch


def test_prepare_staging_renames_checkpoints_and_rejects_hash_mismatch(tmp_path):
    setup = write_release_inputs(tmp_path)
    publisher = setup["publisher"]
    staging = tmp_path / "hub"
    staged = publisher.prepare_staging(
        setup["importance"],
        setup["human"],
        setup["card"],
        staging,
        expected=setup["expected"],
        images_dir=setup["pictures"],
        read_metadata=setup["read_metadata"],
    )
    assert (staging / "focalnet.pt").read_bytes() == b"importance-pt"
    assert (staging / "focalnet-human.pt").read_bytes() == b"human-pt"
    assert (staging / "images" / "example.jpg").read_bytes() == b"jpeg-bytes"
    assert staged["focalnet.onnx"] == setup["expected"]["focalnet.onnx"]
    assert staged["focalnet-human.pt"] == setup["expected"]["focalnet-human.pt"]
    assert staged["images/example.jpg"] == publisher.sha256_file(setup["pictures"] / "example.jpg")

    (setup["human"] / "focalnet-human.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        publisher.prepare_staging(
            setup["importance"],
            setup["human"],
            setup["card"],
            staging,
            expected=setup["expected"],
            images_dir=setup["pictures"],
            read_metadata=setup["read_metadata"],
        )


def test_prepare_staging_rejects_swapped_card_hashes(tmp_path):
    setup = write_release_inputs(tmp_path)
    names = list(setup["expected"])
    swapped = dict(setup["expected"])
    swapped[names[0]], swapped[names[2]] = swapped[names[2]], swapped[names[0]]
    setup["card"].write_text(
        "\n".join(
            [
                *(f"`{name}` `{digest}`" for name, digest in swapped.items()),
                "![example](images/example.jpg)",
            ]
        )
    )
    with pytest.raises(ValueError, match="does not pair"):
        setup["publisher"].prepare_staging(
            setup["importance"],
            setup["human"],
            setup["card"],
            tmp_path / "hub",
            expected=setup["expected"],
            images_dir=setup["pictures"],
            read_metadata=setup["read_metadata"],
        )


def test_prepare_staging_rejects_missing_images_and_stale_sidecars(tmp_path):
    setup = write_release_inputs(tmp_path)
    publisher = setup["publisher"]
    staging = tmp_path / "hub"
    with pytest.raises(FileNotFoundError, match="Missing image directory"):
        publisher.prepare_staging(
            setup["importance"],
            setup["human"],
            setup["card"],
            staging,
            expected=setup["expected"],
            images_dir=tmp_path / "absent-images",
            read_metadata=setup["read_metadata"],
        )

    (setup["pictures"] / "orphan.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="not referenced"):
        publisher.prepare_staging(
            setup["importance"],
            setup["human"],
            setup["card"],
            staging,
            expected=setup["expected"],
            images_dir=setup["pictures"],
            read_metadata=setup["read_metadata"],
        )
    (setup["pictures"] / "orphan.png").unlink()

    (setup["human"] / "focalnet-human.onnx.json").write_text('{"format_version": 1}')
    with pytest.raises(ValueError, match="does not match embedded metadata"):
        publisher.prepare_staging(
            setup["importance"],
            setup["human"],
            setup["card"],
            staging,
            expected=setup["expected"],
            images_dir=setup["pictures"],
            read_metadata=setup["read_metadata"],
        )


def test_renderer_uses_a_font_from_the_environment():
    renderer = load_script(RENDER, "render_hub_examples")
    regular = renderer.resolve_font(*renderer.REGULAR_FONTS)
    bold = renderer.resolve_font(*renderer.BOLD_FONTS)
    assert regular.is_file()
    assert bold.is_file()
    with pytest.raises(FileNotFoundError, match="No usable TTF"):
        renderer.resolve_font(Path("/tmp/missing-focalnet-font.ttf"))
