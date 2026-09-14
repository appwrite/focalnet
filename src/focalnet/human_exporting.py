"""Verified ONNX export for the human crop-ranking model."""

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch

from focalnet.candidates import MAX_CANDIDATES
from focalnet.data import sha256_file
from focalnet.imaging import INPUT_SIZE, MAP_SIZE, MEAN, STD
from focalnet.ranking import HumanCropInference, load_human_checkpoint
from focalnet.teacher import cpu_session


def export_human_model(checkpoint_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    model, checkpoint = load_human_checkpoint(checkpoint_path)
    inference = HumanCropInference(model).eval()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(17)
    image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)
    boxes = torch.zeros(1, MAX_CANDIDATES, 4)
    left_top = torch.rand(1, MAX_CANDIDATES, 2, generator=generator) * 0.3
    size = 0.5 + torch.rand(1, MAX_CANDIDATES, 2, generator=generator) * 0.2
    boxes[..., :2] = left_top
    boxes[..., 2:] = torch.minimum(left_top + size, torch.ones_like(left_top))
    content = torch.tensor([[0.0, 0.21875, 1.0, 0.78125]])
    with torch.inference_mode():
        expected = inference(image, boxes, content)
    metadata = {
        "format_version": 2,
        "task": "human-crop-ranking",
        "input": {
            "image": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "boxes": [1, MAX_CANDIDATES, 4],
            "content": [1, 4],
            "box_coordinates": "normalized oriented source image",
            "content_coordinates": "normalized letterboxed model input",
            "channels": "RGB",
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
            "resize": "bilinear-letterbox",
        },
        "output": {
            "importance": {
                "shape": [1, 1, MAP_SIZE, MAP_SIZE],
                "activation": "sigmoid already applied",
            },
            "crop_scores": {
                "shape": [1, MAX_CANDIDATES],
                "semantics": "relative human-preference logits; compare only within one image",
            },
        },
        "model_config": model.base.config.to_dict(),
        "rank_config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "ranking_parameters": model.ranking_parameter_count(),
        "trained_epochs": checkpoint["epoch"] + 1,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "base_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
        "initialize_checkpoint_sha256": checkpoint.get("initialize_checkpoint_sha256"),
        "dataset_name": checkpoint.get("dataset_name"),
        "dataset_fingerprint": checkpoint["dataset_fingerprint"],
        "candidate_policy": {
            "positions": 5,
            "scales": [1.0, 0.9, 0.8, 0.7, 0.65],
            "retention_tolerance": 0.05,
        },
        "onnxruntime_verification": {
            "provider": "CPUExecutionProvider",
            "graph_optimization": "ORT_DISABLE_ALL",
            "atol": 0.05,
            "rtol": 0.001,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="focalnet-human-export-", dir=output.parent
    ) as directory:
        temporary = Path(directory) / "model.onnx"
        torch.onnx.export(
            inference,
            (image, boxes, content),
            str(temporary),
            input_names=["image", "boxes", "content"],
            output_names=["importance", "crop_scores"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        graph = onnx.load(temporary)
        onnx.helper.set_model_props(graph, {"focalnet": json.dumps(metadata)})
        onnx.checker.check_model(graph)
        onnx.save(graph, temporary)
        session = cpu_session(temporary)
        actual = session.run(
            ["importance", "crop_scores"],
            {"image": image.numpy(), "boxes": boxes.numpy(), "content": content.numpy()},
        )
        errors = []
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(observed, reference.numpy(), atol=5e-2, rtol=1e-3)
            errors.append(float(np.max(np.abs(observed - reference.numpy()))))
        del session
        metadata["export_max_abs_error"] = max(errors)
        onnx.helper.set_model_props(graph, {"focalnet": json.dumps(metadata)})
        onnx.save(graph, temporary)
        temporary.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "model": str(output),
        "bytes": output.stat().st_size,
        "max_abs_error": max(errors),
        "parameters": model.parameter_count(),
        "ranking_parameters": model.ranking_parameter_count(),
    }
