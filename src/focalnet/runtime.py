"""CPU inference without PyTorch, timm, or either teacher model."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from focalnet.crop import focal_point, solve_crop
from focalnet.imaging import INPUT_SIZE, MAP_SIZE, open_image, prepare_image, restore_map
from focalnet.teacher import cpu_session


class Predictor:
    def __init__(self, model: str | Path, threads: int = 1):
        self.session = cpu_session(model, threads)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "image"
            or inputs[0].shape != [1, 3, INPUT_SIZE, INPUT_SIZE]
            or inputs[0].type != "tensor(float)"
        ):
            raise ValueError("Expected FocalNet float32 input 'image' [1,3,256,256]")
        if (
            len(outputs) != 1
            or outputs[0].name != "importance"
            or outputs[0].shape != [1, 1, MAP_SIZE, MAP_SIZE]
        ):
            raise ValueError("Expected FocalNet output 'importance' [1,1,64,64]")
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.metadata = json.loads(metadata.get("focalnet", "{}"))
        if self.metadata.get("format_version") != 1:
            raise ValueError("Missing or incompatible FocalNet metadata; use focalnet export")

    def heatmap(self, image: Image.Image) -> np.ndarray:
        tensor, box = prepare_image(image)
        result = self.session.run(["importance"], {"image": tensor})[0][0, 0]
        if not np.isfinite(result).all() or result.min() < -1e-5 or result.max() > 1.00001:
            raise ValueError("Model produced an invalid importance map")
        return restore_map(np.clip(result, 0, 1), box)

    def predict(
        self, path: str | Path, aspect_ratio: float | None = None
    ) -> tuple[dict, np.ndarray]:
        image = open_image(path)
        heatmap = self.heatmap(image)
        result = {
            "gravity": focal_point(heatmap),
            "importance_peak": float(heatmap.max()),
            "image": {"width": image.width, "height": image.height},
        }
        if aspect_ratio is not None:
            result["crop"] = solve_crop(heatmap, image.width, image.height, aspect_ratio).to_dict()
        return result, heatmap
