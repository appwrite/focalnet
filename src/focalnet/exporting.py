"""Verified FP32 export and representative-data static INT8 quantization."""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from timm.utils import reparameterize_model

from focalnet.data import read_manifest, sha256_file
from focalnet.imaging import INPUT_SIZE, MAP_SIZE, MEAN, STD, open_image, prepare_image
from focalnet.model import InferenceModel, load_checkpoint
from focalnet.runtime import Predictor
from focalnet.teacher import cpu_session


def export_model(checkpoint_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    model, checkpoint = load_checkpoint(str(checkpoint_path))
    model.eval()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(7)
    samples = [
        torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),
        torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, generator=generator),
    ]
    with torch.inference_mode():
        references = [model(sample).sigmoid().numpy() for sample in samples]
        inference_model = InferenceModel(reparameterize_model(model)).eval()
        inference_references = [inference_model(sample).numpy() for sample in samples]
        fusion_errors = [
            float(np.max(np.abs(actual - expected)))
            for actual, expected in zip(inference_references, references, strict=True)
        ]
        reparameterized = all(
            np.allclose(actual, expected, atol=5e-3, rtol=1e-3)
            for actual, expected in zip(inference_references, references, strict=True)
        )
        if not reparameterized:
            # Trained BatchNorm statistics can make RepViT branch folding exceed the strict
            # probability-space parity bound. Export the original eval graph in that case.
            inference_model = InferenceModel(model).eval()
            inference_references = [inference_model(sample).numpy() for sample in samples]
            for actual, expected in zip(inference_references, references, strict=True):
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    metadata = {
        "format_version": 1,
        "input": {
            "name": "image",
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "dtype": "float32",
            "channels": "RGB",
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
            "resize": "bilinear-letterbox",
            "padding": 0,
            "orientation": "apply EXIF before resizing",
            "alpha": "composite on black",
        },
        "output": {
            "name": "importance",
            "shape": [1, 1, MAP_SIZE, MAP_SIZE],
            "activation": "sigmoid already applied",
            "coordinates": "letterboxed-input",
        },
        "model_config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "trained_epochs": checkpoint["epoch"] + 1,
        "reparameterized": reparameterized,
        "fusion_max_abs_error": max(fusion_errors),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": checkpoint.get("manifest_sha256", {}),
        "onnxruntime_verification": {
            "provider": "CPUExecutionProvider",
            "graph_optimization": "ORT_DISABLE_ALL",
            "atol": 0.05,
            "rtol": 0.001,
        },
        "confidence": "No calibrated confidence head; importance_peak is an activation statistic.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="focalnet-export-", dir=output.parent) as directory:
        temporary = Path(directory) / "model.onnx"
        torch.onnx.export(
            inference_model,
            (samples[0],),
            str(temporary),
            input_names=["image"],
            output_names=["importance"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        graph = onnx.load(temporary)
        onnx.helper.set_model_props(graph, {"focalnet": json.dumps(metadata)})
        onnx.checker.check_model(graph)
        onnx.save(graph, temporary)
        session = cpu_session(temporary)
        errors = []
        for sample, expected in zip(samples, inference_references, strict=True):
            actual = session.run(["importance"], {"image": sample.numpy()})[0]
            # Bound probability drift with aggressive platform-specific layout rewrites disabled.
            np.testing.assert_allclose(actual, expected, atol=5e-2, rtol=1e-3)
            errors.append(float(np.max(np.abs(actual - expected))))
        del session
        metadata["export_max_abs_error"] = max(errors)
        onnx.helper.set_model_props(graph, {"focalnet": json.dumps(metadata)})
        onnx.save(graph, temporary)
        temporary.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "model": str(output),
        "bytes": output.stat().st_size,
        "fusion_max_abs_error": max(fusion_errors),
        "max_abs_error": max(errors),
        "reparameterized": reparameterized,
        "parameters": model.parameter_count(),
    }


class _CalibrationReader(CalibrationDataReader):
    def __init__(self, records: list[dict]):
        self.records = iter(records)

    def get_next(self) -> dict | None:
        record = next(self.records, None)
        if record is None:
            return None
        tensor, _ = prepare_image(open_image(record["image"]))
        return {"image": tensor}


def quantize_model(
    model: Path,
    manifest: Path,
    output: Path,
    *,
    samples: int = 128,
    seed: int = 42,
    activation: str = "s8",
) -> dict:
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    if samples <= 0:
        raise ValueError("Calibration samples must be positive")
    activation_types = {"s8": QuantType.QInt8, "u8": QuantType.QUInt8}
    if activation not in activation_types:
        raise ValueError(f"Unsupported activation quantization: {activation}")
    records = read_manifest(manifest)
    random.Random(seed).shuffle(records)
    records = records[:samples]
    # Validate the contract before attempting to quantize an arbitrary graph.
    metadata = Predictor(model).metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="focalnet-int8-", dir=output.parent) as directory:
        prepared = Path(directory) / "prepared.onnx"
        quantized = Path(directory) / "quantized.onnx"
        quant_pre_process(str(model), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            str(prepared),
            str(quantized),
            _CalibrationReader(records),
            quant_format=QuantFormat.QDQ,
            activation_type=activation_types[activation],
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv", "MatMul"],
        )
        metadata["quantization"] = {
            "format": "QDQ",
            "activation": activation.upper(),
            "weights": "S8 per-channel",
            "calibration": "MinMax",
            "samples": len(records),
            "seed": seed,
            "manifest_sha256": sha256_file(manifest),
            "fp32_sha256": sha256_file(model),
            "groups": [record["group"] for record in records],
        }
        graph = onnx.load(quantized)
        onnx.helper.set_model_props(graph, {"focalnet": json.dumps(metadata)})
        onnx.checker.check_model(graph)
        onnx.save(graph, quantized)
        # A runnable graph is necessary; quality requires a separate held-out evaluation.
        predictor = Predictor(quantized)
        predictor.predict(records[0]["image"])
        del predictor
        quantized.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "model": str(output),
        "bytes": output.stat().st_size,
        "calibration_samples": len(records),
    }
