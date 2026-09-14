"""CPU inference for the human-preference crop-ranking model."""

import json
from pathlib import Path

import numpy as np

from focalnet.candidates import MAX_CANDIDATES, generate_candidates, letterbox_content
from focalnet.crop import Crop, focal_point, score_crop
from focalnet.imaging import INPUT_SIZE, MAP_SIZE, open_image, prepare_image, restore_map
from focalnet.teacher import cpu_session


class HumanCropPredictor:
    def __init__(self, model: str | Path, threads: int = 1):
        self.session = cpu_session(model, threads)
        inputs = {value.name: value for value in self.session.get_inputs()}
        outputs = {value.name: value for value in self.session.get_outputs()}
        expected_inputs = {
            "image": ([1, 3, INPUT_SIZE, INPUT_SIZE], "tensor(float)"),
            "boxes": ([1, MAX_CANDIDATES, 4], "tensor(float)"),
            "content": ([1, 4], "tensor(float)"),
        }
        expected_outputs = {
            "importance": [1, 1, MAP_SIZE, MAP_SIZE],
            "crop_scores": [1, MAX_CANDIDATES],
        }
        if set(inputs) != set(expected_inputs) or any(
            inputs[name].shape != shape or inputs[name].type != dtype
            for name, (shape, dtype) in expected_inputs.items()
        ):
            raise ValueError("Human crop model has an incompatible input contract")
        if set(outputs) != set(expected_outputs) or any(
            outputs[name].shape != shape for name, shape in expected_outputs.items()
        ):
            raise ValueError("Human crop model has an incompatible output contract")
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.metadata = json.loads(metadata.get("focalnet", "{}"))
        if (
            self.metadata.get("format_version") != 2
            or self.metadata.get("task") != "human-crop-ranking"
        ):
            raise ValueError("Missing or incompatible human crop metadata")

    @staticmethod
    def _pixel_crop(box: np.ndarray, width: int, height: int) -> Crop:
        left = max(0, min(width - 1, int(box[0] * width + 0.5)))
        top = max(0, min(height - 1, int(box[1] * height + 0.5)))
        right = max(left + 1, min(width, int(box[2] * width + 0.5)))
        bottom = max(top + 1, min(height, int(box[3] * height + 0.5)))
        return Crop(left, top, right - left, bottom - top, 0)

    def predict(
        self,
        path: str | Path,
        aspect_ratio: float,
        *,
        retention_tolerance: float = 0.05,
    ) -> tuple[dict, np.ndarray]:
        if not 0 <= retention_tolerance <= 1:
            raise ValueError("Retention tolerance must be in [0, 1]")
        image = open_image(path)
        tensor, content_box = prepare_image(image)
        candidates = generate_candidates(image.width, image.height, aspect_ratio)
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError(f"Generated {len(candidates)} crops; maximum is {MAX_CANDIDATES}")
        boxes = np.zeros((1, MAX_CANDIDATES, 4), dtype=np.float32)
        boxes[0, : len(candidates)] = candidates
        content = letterbox_content(content_box)[None]
        importance, scores = self.session.run(
            ["importance", "crop_scores"],
            {"image": tensor, "boxes": boxes, "content": content},
        )
        if (
            not np.isfinite(importance).all()
            or importance.min() < -1e-5
            or importance.max() > 1.00001
            or not np.isfinite(scores[0, : len(candidates)]).all()
        ):
            raise ValueError("Model produced invalid importance or crop scores")
        importance = np.clip(importance, 0, 1)
        heatmap = restore_map(importance[0, 0], content_box)
        pixel_crops = [
            self._pixel_crop(candidate, image.width, image.height) for candidate in candidates
        ]
        retention = np.asarray(
            [score_crop(heatmap, crop, image.width, image.height) for crop in pixel_crops]
        )
        maximum_retention = float(retention.max())
        eligible = retention >= maximum_retention - retention_tolerance
        eligible_indices = np.flatnonzero(eligible)
        selected = int(eligible_indices[np.argmax(scores[0, eligible_indices])])
        crop = pixel_crops[selected]
        crop = Crop(crop.left, crop.top, crop.width, crop.height, float(retention[selected]))
        return (
            {
                "gravity": focal_point(heatmap),
                "importance_peak": float(heatmap.max()),
                "image": {"width": image.width, "height": image.height},
                "crop": crop.to_dict(),
                "human_score": float(scores[0, selected]),
                "candidate_count": len(candidates),
                "retention_regret": maximum_retention - float(retention[selected]),
            },
            heatmap,
        )
