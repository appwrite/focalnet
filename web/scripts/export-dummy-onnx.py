"""Write a contract-compatible ONNX file when the Modal volume is unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def constant(name: str, array: np.ndarray):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name))


def export_dummy(output: Path) -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    importance = np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2 * 12.0**2)).astype(np.float32)[
        None, None
    ]
    scores = np.linspace(1.0, 0.2, 128, dtype=np.float32)[None]
    graph = helper.make_graph(
        [
            constant("importance_const", importance),
            constant("scores_const", scores),
            constant("zero", np.array(0, dtype=np.float32)),
            constant("zero_row", np.zeros((1, 128), dtype=np.float32)),
            constant("axes_2", np.array([2], dtype=np.int64)),
            helper.make_node("ReduceSum", ["image"], ["image_mass"], keepdims=0),
            helper.make_node("Mul", ["image_mass", "zero"], ["image_scale"]),
            helper.make_node("Add", ["importance_const", "image_scale"], ["importance"]),
            helper.make_node("ReduceSum", ["boxes", "axes_2"], ["box_mass"], keepdims=0),
            helper.make_node("ReduceSum", ["content"], ["content_mass"], keepdims=0),
            helper.make_node("Mul", ["box_mass", "zero_row"], ["box_scale"]),
            helper.make_node("Mul", ["content_mass", "zero_row"], ["content_scale"]),
            helper.make_node("Add", ["box_scale", "content_scale"], ["input_scale"]),
            helper.make_node("Add", ["scores_const", "input_scale"], ["crop_scores"]),
        ],
        "focalnet-dummy",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 256, 256]),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 128, 4]),
            helper.make_tensor_value_info("content", TensorProto.FLOAT, [1, 4]),
        ],
        [
            helper.make_tensor_value_info("importance", TensorProto.FLOAT, [1, 1, 64, 64]),
            helper.make_tensor_value_info("crop_scores", TensorProto.FLOAT, [1, 128]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.helper.set_model_props(
        model,
        {
            "focalnet": json.dumps(
                {
                    "format_version": 2,
                    "task": "human-crop-ranking",
                    "placeholder": True,
                }
            )
        },
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == "__main__":
    export_dummy(Path("web/public/focalnet-human.onnx"))
