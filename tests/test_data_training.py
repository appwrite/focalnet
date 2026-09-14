import json

import numpy as np
import pytest
import torch
from PIL import Image

from focalnet.data import label_images, read_manifest, split_manifest, write_manifest
from focalnet.imaging import Letterbox, restore_map
from focalnet.model import ModelConfig
from focalnet.teacher import TeacherConfig
from focalnet.training import (
    ImportanceDataset,
    TrainConfig,
    ensure_disjoint,
    importance_loss,
    train,
)


def make_records(directory, count=4):
    records = []
    for index in range(count):
        image = directory / f"{index}.png"
        target = directory / f"{index}.npy"
        Image.new("RGB", (103, 200), (100, 150, 200)).save(image)
        values = np.zeros((64, 64), np.float32)
        values[16:32, 8:24] = 1
        np.save(target, values)
        records.append({"image": str(image), "importance": str(target), "group": str(index // 2)})
    return records


def test_split_groups_related_images_and_resolves_paths(tmp_path):
    records = make_records(tmp_path, 8)
    source = tmp_path / "source" / "labels.jsonl"
    write_manifest(source, records)
    output = tmp_path / "splits"
    split_manifest(source, output, 0.25, 42)
    train, val = read_manifest(output / "train.jsonl"), read_manifest(output / "val.jsonl")
    assert len(train) == 6 and len(val) == 2
    ensure_disjoint(train, val)
    assert all(record["image"].startswith(str(tmp_path)) for record in train + val)
    with pytest.raises(ValueError, match="leakage"):
        ensure_disjoint(train, train[:1])


def test_manifest_written_through_symlink_resolves_against_real_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    records = make_records(real)
    manifest = alias / "split" / "train.jsonl"
    write_manifest(manifest, records)
    loaded = read_manifest(manifest)
    assert [record["image"] for record in loaded] == [record["image"] for record in records]


def test_dataset_horizontal_flip_keeps_image_and_target_coordinates(tmp_path, monkeypatch):
    records = make_records(tmp_path)
    monkeypatch.setattr("focalnet.training.random.random", lambda: 0.0)
    monkeypatch.setattr("focalnet.training.random.uniform", lambda *_: 1.0)
    _, target, valid = ImportanceDataset(records, augment=True)[0]
    source = restore_map(target[0].numpy(), Letterbox.fit(103, 200))
    assert source[:, 40:56].sum() > 0.95 * source.sum()
    assert torch.all(target[valid == 0] == 0)


def test_loss_ignores_padding_and_handles_empty_teacher():
    logits = torch.zeros(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 24:40, 24:40] = 1
    valid = torch.zeros_like(logits)
    valid[:, :, 16:48] = 1
    loss, _ = importance_loss(logits, target, valid)
    changed = logits.detach().clone()
    changed[valid == 0] = 100
    same_loss, _ = importance_loss(changed, target, valid)
    assert loss.item() == same_loss.item()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[valid == 0] == 0)
    assert logits.grad[0, 0, 30, 30] < 0
    assert logits.grad[1, 0, 30, 30] > 0


def test_manifest_requires_explicit_group(tmp_path):
    record = make_records(tmp_path)[0]
    del record["group"]
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="group"):
        read_manifest(manifest)


def test_label_images_accepts_an_explicit_subset_without_scanning(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    selected = images / "selected.jpg"
    ignored = images / "ignored.jpg"
    Image.new("RGB", (20, 10), (20, 40, 60)).save(selected)
    Image.new("RGB", (20, 10), (60, 40, 20)).save(ignored)
    saliency_model = tmp_path / "u2net.onnx"
    face_model = tmp_path / "yunet.onnx"
    saliency_model.write_bytes(b"saliency")
    face_model.write_bytes(b"face")

    class FakeTeacher:
        def __init__(self, *_args, **_kwargs):
            pass

        def predict(self, _image):
            return np.ones((64, 64), np.float32), {"faces": []}

    monkeypatch.setattr("focalnet.data.Teacher", FakeTeacher)
    monkeypatch.setattr(
        "focalnet.data.image_paths",
        lambda *_: pytest.fail("explicit source paths must bypass directory traversal"),
    )
    output = tmp_path / "labels"
    result = label_images(
        images,
        output,
        saliency_model,
        face_model,
        TeacherConfig(),
        source_paths=[selected],
    )
    records = read_manifest(output / "labels.jsonl")
    assert result["total"] == 1
    assert [record["image"] for record in records] == [str(selected)]

    with pytest.raises(ValueError, match="escapes image directory"):
        label_images(
            images,
            tmp_path / "invalid-labels",
            saliency_model,
            face_model,
            TeacherConfig(),
            source_paths=[images / ".." / "outside.jpg"],
        )


@pytest.mark.integration
def test_interrupted_training_resumes_with_identical_cpu_result(tmp_path, monkeypatch):
    import focalnet.training as training

    records = make_records(tmp_path)
    train_path, val_path = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_manifest(train_path, records[:2])
    write_manifest(val_path, records[2:])
    config = TrainConfig(
        epochs=2, batch_size=2, freeze_encoder_epochs=0, pretrained=False, threads=1, device="cpu"
    )
    model_config = ModelConfig("mobilenetv4_conv_small")
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    train(train_path, val_path, complete, model_config, config)
    save = training._save_checkpoint

    def interrupt_after_first_epoch(path, checkpoint):
        save(path, checkpoint)
        if path.name == "best.pt" and checkpoint["epoch"] == 0:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(training, "_save_checkpoint", interrupt_after_first_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(train_path, val_path, interrupted, model_config, config)
    train(train_path, val_path, interrupted, model_config, config, interrupted / "last.pt")
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert actual["epoch"] == 1
    assert actual["history"] == expected["history"]
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, atol=0, rtol=0)
