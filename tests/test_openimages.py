import csv
import json
from io import BytesIO

import pytest
from PIL import Image

from focalnet.openimages import (
    AcquireConfig,
    _choose_bucket,
    _normalise_jpeg,
    _quotas,
    attach_metadata,
    plan_candidates,
)


def csv_file(path, rows):
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path.as_uri()


def test_quotas_add_up_and_prioritize_people():
    assert sum(_quotas(100_003).values()) == 100_003
    assert sum(_quotas(400_003, "expanded").values()) == 400_003
    assert _quotas(400_000, "expanded")["food"] == 44_000
    mapping = {"person": "human", "dog": "animal"}
    assert _choose_bucket({"person", "dog"}, 20, mapping) == "human"
    assert _choose_bucket(set(), 6, mapping) == "busy"
    assert _choose_bucket(set(), 5, mapping) == "general"


def test_plan_and_metadata_are_deterministic_and_keep_attribution(tmp_path):
    classes = csv_file(
        tmp_path / "classes.csv",
        [
            ["person", "Person"],
            ["dog", "Dog"],
            ["food", "Food"],
            ["car", "Car"],
            ["tree", "Tree"],
        ],
    )
    boxes = [["ImageID", "LabelName"]]
    for index in range(800):
        image_id = f"{index:016x}"
        if index < 150:
            label, box_count = "person", 1
        elif index < 250:
            label, box_count = "dog", 1
        elif index < 350:
            label, box_count = "food", 1
        elif index < 450:
            label, box_count = "car", 1
        elif index < 550:
            label, box_count = "tree", 1
        elif index < 650:
            label, box_count = "other", 6
        else:
            label, box_count = "other", 1
        boxes.extend([[image_id, label]] * box_count)
    box_url = csv_file(tmp_path / "boxes.csv", boxes)
    candidates = tmp_path / "candidates.jsonl"
    report = plan_candidates(
        candidates,
        AcquireConfig(count=100, seed=7),
        class_url=classes,
        box_url=box_url,
        exclude_image_ids={f"{index:016x}" for index in range(10)},
    )
    assert report["scanned"] == 800
    records = [json.loads(line) for line in candidates.read_text().splitlines()]
    assert len({record["image_id"] for record in records}) == len(records)
    selected = {record["image_id"] for record in records}
    assert not selected & {f"{index:016x}" for index in range(10)}
    metadata_rows = [
        [
            "ImageID",
            "OriginalURL",
            "OriginalLandingURL",
            "License",
            "Author",
            "AuthorProfileURL",
            "Title",
            "Rotation",
        ]
    ]
    for image_id in selected:
        metadata_rows.append(
            [
                image_id,
                "original",
                "landing",
                "https://creativecommons.org/licenses/by/2.0/",
                "author",
                "profile",
                "title",
                "90",
            ]
        )
    metadata = csv_file(tmp_path / "metadata.csv", metadata_rows)
    enriched = tmp_path / "enriched.jsonl"
    result = attach_metadata(candidates, enriched, metadata_url=metadata)
    assert result["with_cc_by_metadata"] == len(records)
    assert json.loads(enriched.read_text().splitlines()[0])["Author"] == "author"


def test_normalise_jpeg_resizes_and_bakes_counterclockwise_rotation():
    source = Image.new("RGB", (100, 50), "black")
    source.paste("red", (0, 0, 50, 50))
    buffer = BytesIO()
    source.save(buffer, "JPEG", quality=100)
    data, width, height = _normalise_jpeg(buffer.getvalue(), "90", 64, 88)
    assert (width, height) == (32, 64)
    with Image.open(BytesIO(data)) as image:
        # Left becomes bottom under a 90-degree counterclockwise rotation.
        assert image.getpixel((16, 48))[0] > image.getpixel((16, 8))[0] + 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"max_side": 0},
        {"workers": 0},
        {"storage_limit_gib": 0},
        {"min_free_gib": 0},
        {"jpeg_quality": 96},
    ],
)
def test_invalid_acquisition_settings_fail(kwargs):
    with pytest.raises(ValueError):
        AcquireConfig(**kwargs)
