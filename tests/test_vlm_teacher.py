import base64
import json as json_module
from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import Image

from focalnet.crop import Crop, score_crop, solve_crop
from focalnet.vlm_teacher import (
    OpenRouterTeacher,
    Subject,
    VlmAnnotation,
    bakeoff_vlm,
    box_coverage,
    encode_data_url,
    label_vlm_images,
    parse_annotation,
    parse_json_content,
    region_importance_target,
)


def dunk_annotation(*, face_only: bool) -> VlmAnnotation:
    if face_only:
        return VlmAnnotation(
            (Subject("face", 1.0, "face", 0.32, 0.28, 0.58, 0.46),),
            (0,),
            (0,),
        )
    action = Subject("dunk", 1.0, "person", 0.16, 0.04, 0.74, 0.96)
    extra = Subject("bench player", 0.12, "person", 0.62, 0.52, 0.98, 0.97)
    return VlmAnnotation((action, extra), (0,), (0,))


def test_parse_strips_markdown_and_rejects_bad_boxes():
    payload = parse_json_content(
        '```json\n{"subjects":[{"label":"ball","importance":0.8,"kind":"object",'
        '"x1":0.1,"y1":0.1,"x2":0.4,"y2":0.5}],"keep_together":[0],"must_contain":[0]}\n```'
    )
    annotation = parse_annotation(payload)
    assert annotation.subjects[0].kind == "object"
    assert annotation.must_contain == (0,)
    swapped = parse_annotation(
        {
            "subjects": [
                {"label": "x", "importance": 1, "x1": 0.9, "y1": 0.1, "x2": 0.2, "y2": 0.3}
            ]
        }
    )
    assert swapped.subjects[0].box == pytest.approx((0.2, 0.1, 0.9, 0.3))
    with pytest.raises(ValueError, match="normalized"):
        parse_annotation(
            {
                "subjects": [
                    {"label": "x", "importance": 1, "x1": 0.5, "y1": 0.1, "x2": 0.5, "y2": 0.3}
                ]
            }
        )
    permille = parse_annotation(
        {
            "subjects": [
                {
                    "label": "dunk",
                    "importance": 1,
                    "kind": "person",
                    "x1": 180,
                    "y1": 30,
                    "x2": 720,
                    "y2": 950,
                }
            ],
            "must_contain": [0],
        }
    )
    assert permille.subjects[0].box == pytest.approx((0.18, 0.03, 0.72, 0.95))
    gemini = parse_annotation(
        {
            "subjects": [
                {
                    "label": "dunking player",
                    "importance": 1.0,
                    "kind": "person",
                    "x1": 305,
                    "y1": 108,
                    "x2": 718,
                    "y2": 964,
                },
                {
                    "label": "basketball and rim",
                    "importance": 1.0,
                    "kind": "object",
                    "x1": 235,
                    "y1": 83,
                    "x2": 671,
                    "y2": 312,
                },
                {
                    "label": "player 13",
                    "importance": 0.7,
                    "kind": "person",
                    "x1": 702,
                    "y1": 548,
                    "x2": 994,
                    "y2": 998,
                },
            ],
            "keep_together": [0, 1, 2],
            "must_contain": [0, 1],
        }
    )
    assert gemini.subjects[0].box == pytest.approx((0.305, 0.108, 0.718, 0.964))
    pixels = parse_annotation(
        {
            "subjects": [
                {
                    "label": "ball",
                    "importance": 1,
                    "kind": "object",
                    "x1": 500,
                    "y1": 300,
                    "x2": 1500,
                    "y2": 1200,
                }
            ],
            "must_contain": [0],
        },
        image_size=(2000, 1500),
    )
    assert pixels.subjects[0].box == pytest.approx((0.25, 0.2, 0.75, 0.8))


def test_action_map_keeps_full_pose_where_face_map_clips():
    width, height = 400, 600
    action = dunk_annotation(face_only=False)
    face = dunk_annotation(face_only=True)
    vlm = region_importance_target(action)
    head = region_importance_target(face)
    tight = Crop(left=120, top=160, width=130, height=130, retained_importance=0)
    assert score_crop(head, tight, width, height) > 0.8
    assert score_crop(vlm, tight, width, height) < 0.35
    vlm_crop = solve_crop(vlm, width, height, 1.0)
    face_crop = solve_crop(head, width, height, 1.0)
    feet = (0.28, 0.82, 0.62, 0.96)
    assert box_coverage(vlm_crop, feet, width, height) > box_coverage(
        face_crop, feet, width, height
    )


def test_encode_data_url_resizes_and_drops_alpha():
    image = Image.new("RGBA", (2000, 1000), (10, 20, 30, 255))
    url = encode_data_url(image, max_side=512)
    assert url.startswith("data:image/jpeg;base64,")
    jpeg = Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert jpeg.mode == "RGB"
    assert max(jpeg.size) == 512


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=self.request, response=self)

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(
            {
                "choices": [{"message": {"content": json_module.dumps(self.payload)}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            }
        )


def test_openrouter_teacher_parses_json_object(tmp_path):
    payload = dunk_annotation(face_only=False).to_dict()
    client = FakeClient(payload)
    teacher = OpenRouterTeacher("sk-test", "google/gemini-3.1-flash-lite", client=client)
    annotation, metadata = teacher.complete(Image.new("RGB", (64, 96), (12, 34, 56)))
    assert annotation.subjects[0].label == "dunk"
    assert metadata["model"] == "google/gemini-3.1-flash-lite"
    assert client.calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_label_and_bakeoff_use_injected_teacher(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (80, 120), (40, 80, 120)).save(images / "dunk.jpg")
    payload = dunk_annotation(face_only=False).to_dict()

    class Scripted:
        def __init__(self):
            self.api_key = "sk-test"
            self.model = "google/gemini-3.1-flash-lite"

        def complete(self, image):
            return parse_annotation(payload), {"model": self.model, "usage": {}}

    output = tmp_path / "labels"
    result = label_vlm_images(images, output, teacher=Scripted())
    assert result["created"] == 1
    manifest = json_module.loads((output / "labels.jsonl").read_text().splitlines()[0])
    assert manifest["annotation"]["subjects"][0]["label"] == "dunk"
    heatmap = np.load(output / manifest["importance"])
    assert heatmap.shape == (64, 64) and heatmap.max() == pytest.approx(1.0)

    expected = {"dunk.jpg": [(0.16, 0.04, 0.74, 0.96)]}
    bakeoff = bakeoff_vlm(
        images,
        ["google/gemini-3.1-flash-lite"],
        expected=expected,
        teacher_factory=lambda model: Scripted(),
    )
    coverage = bakeoff["models"][0]["images"][0]["crops"][str(4 / 5)]["min_box_coverage"]
    assert coverage > 0.85
