import numpy as np
import pytest

from focalnet.imaging import Letterbox
from focalnet.teacher import Face, TeacherConfig, decode_faces, importance_target


def test_small_face_has_priority_over_large_body_by_total_mass():
    saliency = np.zeros((64, 64))
    saliency[32:, 8:56] = 1
    target = importance_target(saliency, [Face(0.45, 0.10, 0.55, 0.20, 0.95)])
    assert target[:24].sum() / target.sum() == pytest.approx(0.75, abs=0.001)
    assert target[32:].sum() / target.sum() == pytest.approx(0.25, abs=0.001)
    assert target.max() == 1


def test_multiple_faces_and_fallback_labels():
    empty = np.zeros((64, 64))
    target = importance_target(
        empty, [Face(0.1, 0.1, 0.2, 0.2, 0.95), Face(0.8, 0.1, 0.9, 0.2, 0.95)]
    )
    assert target[:, :24].sum() / target.sum() == pytest.approx(0.5, abs=0.001)
    assert target[:, 40:].sum() / target.sum() == pytest.approx(0.5, abs=0.001)
    np.testing.assert_array_equal(importance_target(empty, []), empty)
    raw = np.eye(64)
    np.testing.assert_array_equal(importance_target(raw, [Face(0.1, 0.1, 0.2, 0.2, 0.81)]), raw)


def test_yunet_decoding_score_nms_and_letterbox_removal():
    outputs = {}
    for stride in (8, 16, 32):
        count = (640 // stride) ** 2
        outputs[f"cls_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, count, 4), np.float32)
    for stride, score in ((8, 0.96), (16, 0.92)):
        index = (320 // stride) * (640 // stride) + 320 // stride
        outputs[f"cls_{stride}"][0, index] = score
        outputs[f"obj_{stride}"][0, index] = score
        outputs[f"bbox_{stride}"][0, index] = [0, 0, np.log(64 / stride), np.log(96 / stride)]
    # A reliable candidate wholly in the top padding must be discarded.
    outputs["cls_8"][0, 1] = 0.99
    outputs["obj_8"][0, 1] = 0.99
    detections = decode_faces(outputs, Letterbox.fit(640, 320, 640), TeacherConfig())
    assert len(detections) == 1
    face = detections[0]
    assert [face.x1, face.y1, face.x2, face.y2] == pytest.approx([0.45, 0.35, 0.55, 0.65])
    assert face.confidence == pytest.approx(0.96)


def test_invalid_face_target_fails():
    with pytest.raises(ValueError, match="normalized box"):
        importance_target(np.ones((64, 64)), [Face(0.9, 0.1, 0.1, 0.2, 0.95)])
