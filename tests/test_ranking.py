import json

import numpy as np
import pytest
import torch
from PIL import Image

from focalnet.candidates import generate_candidates, letterbox_content
from focalnet.cpc import CPCDataset, cpc_paths, read_cpc_annotation
from focalnet.human_exporting import export_human_model
from focalnet.human_runtime import HumanCropPredictor
from focalnet.human_training import (
    GAICDataset,
    collate_crops,
    crop_ranking_loss,
    ranking_metrics,
    read_gaic_annotation,
)
from focalnet.imaging import Letterbox
from focalnet.model import FocalNet, ModelConfig
from focalnet.ranking import HumanCropNet


@pytest.mark.parametrize("ratio", [1.0, 16 / 9, 4 / 5])
def test_candidates_have_requested_pixel_ratio_and_valid_bounds(ratio):
    boxes = generate_candidates(1600, 900, ratio)
    assert 1 < len(boxes) <= 125
    assert np.all((boxes >= 0) & (boxes <= 1))
    assert np.all(boxes[:, 2:] > boxes[:, :2])
    pixel_ratios = (boxes[:, 2] - boxes[:, 0]) * 1600 / ((boxes[:, 3] - boxes[:, 1]) * 900)
    np.testing.assert_allclose(pixel_ratios, ratio, rtol=1e-5, atol=1e-6)
    assert np.unique(boxes, axis=0).shape == boxes.shape


def test_ranker_scores_variable_candidate_lists_without_changing_importance_head():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    base = FocalNet(ModelConfig("mobilenetv4_conv_small"))
    model = HumanCropNet(base).eval()
    images = torch.randn(2, 3, 256, 256)
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.8, 1.0], [0.2, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 0.9], [0.1, 0.1, 0.9, 0.9]],
        ]
    )
    content = torch.from_numpy(
        np.stack(
            [
                letterbox_content(Letterbox.fit(1600, 900)),
                letterbox_content(Letterbox.fit(800, 1200)),
            ]
        )
    )
    with torch.inference_mode():
        expected = base(images)
        importance, scores = model(images, boxes, content)
    torch.testing.assert_close(importance, expected)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()
    assert model.ranking_parameter_count() < 20_000


def test_ranking_parameters_receive_gradients_while_frozen_base_does_not():
    torch.set_num_threads(1)
    model = HumanCropNet(FocalNet(ModelConfig("mobilenetv4_conv_small")))
    model.base.requires_grad_(False)
    images = torch.randn(2, 3, 256, 256)
    boxes = torch.tensor([[[0.0, 0.0, 0.8, 1.0], [0.2, 0.0, 1.0, 1.0]]] * 2, dtype=torch.float32)
    content = torch.tensor([[0.0, 0.2, 1.0, 0.8]] * 2)
    _, scores = model(images, boxes, content)
    scores.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for module in (model.rank_projection, model.rank_head)
        for parameter in module.parameters()
    )


def test_gaic_annotation_coordinates_and_invalid_scores(tmp_path):
    annotation = tmp_path / "crop.txt"
    annotation.write_text("10 20 90 180 4.5\n0 0 100 200 -2\n20 30 80 150 2.0\n")
    boxes, scores = read_gaic_annotation(annotation, 200, 100)
    np.testing.assert_allclose(boxes, [[0.1, 0.1, 0.9, 0.9], [0.15, 0.2, 0.75, 0.8]])
    np.testing.assert_allclose(scores, [4.5, 2.0])


def test_gaic_dataset_and_padding_collate(tmp_path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "annotations" / split).mkdir(parents=True)
    for split, name, lines in (
        ("train", "one", "0 0 80 120 4\n10 10 70 100 2\n"),
        ("train", "two", "0 0 60 100 3\n5 5 55 95 2\n10 10 50 90 1\n"),
        ("val", "three", "0 0 50 90 4\n5 5 45 85 2\n"),
    ):
        Image.new("RGB", (120 if name == "one" else 100, 80 if name == "one" else 60)).save(
            tmp_path / "images" / split / f"{name}.jpg"
        )
        (tmp_path / "annotations" / split / f"{name}.txt").write_text(lines)
    dataset = GAICDataset(tmp_path, "train")
    batch = collate_crops([dataset[0], dataset[1]])
    images, boxes, scores, valid, content, groups = batch
    assert images.shape == (2, 3, 256, 256)
    assert boxes.shape == (2, 3, 4)
    assert scores.shape == valid.shape == (2, 3)
    assert valid.sum().item() == 5
    assert content.shape == (2, 4)
    assert groups == ("one", "two")


def test_cpc_ratings_coordinates_and_deterministic_split(tmp_path):
    images = tmp_path / "images" / "all_images"
    annotations = tmp_path / "CollectedAnnotationsRaw" / "all_labels"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    payload = {
        "scores": [[1, 0, 1], [1, 1, 0]],
        "bboxes": [[10, 0, 90, 80], [0, 8, 100, 72], [20, 10, 80, 70]],
    }
    for index in range(10):
        name = f"image-{index}.jpg"
        Image.new("RGB", (100, 80), (index, 20, 30)).save(images / name)
        encoded = json.dumps(json.dumps(payload)) if index == 0 else json.dumps(payload)
        (annotations / f"{name}.txt").write_text(encoded)
    boxes, scores = read_cpc_annotation(annotations / "image-0.jpg.txt", 100, 80)
    np.testing.assert_allclose(
        boxes, [[0.1, 0, 0.9, 1], [0, 0.1, 1, 0.9], [0.2, 0.125, 0.8, 0.875]]
    )
    np.testing.assert_allclose(scores, [1, 0.5, 0.5])
    train = cpc_paths(tmp_path, "train", seed=7)
    val = cpc_paths(tmp_path, "val", seed=7)
    assert len(train) == 9 and len(val) == 1
    assert set(train).isdisjoint(val)
    dataset = CPCDataset(tmp_path, "train", seed=7)
    assert len(dataset) == 9
    assert dataset[0][1].shape == (3, 4)


def test_pairwise_loss_rewards_the_human_ordering():
    target = torch.tensor([[1.0, 3.0, 5.0], [4.0, 2.0, 0.0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    good = torch.tensor([[-2.0, 0.0, 2.0], [1.0, -1.0, 99.0]], requires_grad=True)
    bad = -good.detach()
    good_loss, parts = crop_ranking_loss(good, target, valid)
    bad_loss, _ = crop_ranking_loss(bad, target, valid)
    assert good_loss < bad_loss
    assert parts["pairwise"] > 0 and parts["regression"] >= 0
    good_loss.backward()
    assert torch.isfinite(good.grad).all()
    assert good.grad[1, 2] == 0


def test_ranking_metrics_are_one_for_exact_order_and_handle_ties():
    target = [np.array([1.0, 2.0, 2.0, 4.0, 5.0]), np.array([3.0, 1.0, 2.0])]
    metrics = ranking_metrics([values.copy() for values in target], target)
    assert metrics["srcc"] == pytest.approx(1)
    assert metrics["pcc"] == pytest.approx(1)
    assert metrics["acc_5"] == pytest.approx(1)
    assert metrics["acc_10"] == pytest.approx(1)
    assert metrics["pairwise_accuracy"] == pytest.approx(1)


@pytest.mark.integration
def test_human_crop_onnx_export_and_runtime(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = HumanCropNet(FocalNet(ModelConfig("mobilenetv4_conv_small"))).eval()
    checkpoint = tmp_path / "human.pt"
    torch.save(
        {
            "format_version": 2,
            "task": "human-crop-ranking",
            "input_size": 256,
            "map_size": 64,
            "model_config": model.base.config.to_dict(),
            "rank_config": model.config.to_dict(),
            "base_checkpoint_sha256": "synthetic",
            "dataset_fingerprint": "synthetic",
            "model": model.state_dict(),
            "epoch": 0,
        },
        checkpoint,
    )
    artifact = tmp_path / "human.onnx"
    report = export_human_model(checkpoint, artifact)
    assert report["max_abs_error"] < 1e-4
    image = tmp_path / "image.jpg"
    Image.new("RGB", (640, 360), (100, 150, 200)).save(image)
    result, heatmap = HumanCropPredictor(artifact).predict(image, 1.0)
    assert heatmap.shape == (64, 64)
    assert result["candidate_count"] <= 128
    assert result["crop"]["width"] == result["crop"]["height"]
    assert result["retention_regret"] <= 0.05 + 1e-8
