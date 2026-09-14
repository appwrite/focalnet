import numpy as np
import pytest
from PIL import Image

from focalnet.imaging import (
    Letterbox,
    open_image,
    prepare_image,
    project_map,
    restore_map,
)


def test_preprocessing_keeps_aspect_alpha_channel_order_and_neutral_padding():
    image = Image.new("RGBA", (200, 100), (200, 100, 50, 128))
    rgb, box = prepare_image(image)
    assert (box.left, box.top, box.width, box.height) == (0, 64, 256, 128)
    assert rgb.shape == (1, 3, 256, 256)
    assert not rgb[0, :, :64].any()
    np.testing.assert_allclose(
        rgb[0, :, 128, 128],
        (np.array([200, 100, 50]) * 128 / 255 / 255 - [0.485, 0.456, 0.406])
        / [0.229, 0.224, 0.225],
        # Pillow's premultiplied-alpha resize can round RGB by one 8-bit value.
        atol=1 / 255 / 0.224,
    )
    bgr, _ = prepare_image(image, 640, bgr=True)
    np.testing.assert_allclose(bgr[0, :, 320, 320], [25, 50, 100], atol=1)
    assert not bgr[0, :, :160].any()


@pytest.mark.parametrize("dimensions", [(199, 301), (640, 320), (1, 10000), (901, 109)])
def test_padding_cannot_affect_restored_heatmap(dimensions):
    box = Letterbox.fit(*dimensions)
    valid = box.coverage()
    heatmap = np.where(valid > 0, 0.4, 1e6).astype(np.float32)
    np.testing.assert_allclose(restore_map(heatmap, box), 0.4, atol=1e-6)
    assert valid.sum() == pytest.approx(box.width * box.height / 16)


def test_letterbox_roundtrip_preserves_source_coordinates():
    source = np.broadcast_to((np.arange(64) + 0.5) / 64, (64, 64)).astype(np.float32)
    box = Letterbox.fit(303, 509)
    projected, valid = project_map(source, box)
    restored = restore_map(projected, box)
    np.testing.assert_allclose(restored[:, 2:-2], source[:, 2:-2], atol=0.012)
    assert np.all(projected[valid == 0] == 0)


def test_exif_orientation_is_applied_before_coordinates(tmp_path):
    image = Image.new("RGB", (8, 4))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    assert open_image(path).size == (4, 8)


def test_rounding_matches_go_half_up():
    assert Letterbox.fit(512, 257).height == 129
