import numpy as np
import pytest

from focalnet.crop import Crop, focal_point, score_crop, solve_crop


def test_multisubject_crop_uses_mass_instead_of_centering_on_centroid():
    heatmap = np.zeros((64, 64))
    heatmap[30, 6], heatmap[30, 57] = 4, 6
    point = focal_point(heatmap)
    assert 0.5 < point["x"] < 0.6
    crop = solve_crop(heatmap, 1000, 500, 1)
    assert crop.left >= 400
    assert crop.retained_importance == pytest.approx(0.6)
    wide = solve_crop(heatmap, 1000, 500, 1.8)
    assert wide.retained_importance == pytest.approx(1)


def test_uniform_and_empty_maps_choose_center():
    uniform = solve_crop(np.ones((8, 8)), 1000, 600, 1)
    assert (uniform.left, uniform.top, uniform.width, uniform.height) == (200, 0, 600, 600)
    assert uniform.retained_importance == pytest.approx(0.6)
    empty = solve_crop(np.zeros((8, 8)), 600, 1000, 1)
    assert (empty.left, empty.top, empty.retained_importance) == (0, 200, 0)
    assert focal_point(np.zeros((8, 8))) == {"x": 0.5, "y": 0.5}


def test_crop_position_is_invariant_to_importance_scale():
    heatmap = np.zeros((8, 8))
    heatmap[2, 6] = 1
    regular = solve_crop(heatmap, 1000, 500, 1)
    tiny = solve_crop(heatmap * 1e-20, 1000, 500, 1)
    assert tiny.left == regular.left
    assert tiny.retained_importance == pytest.approx(regular.retained_importance)


def test_crop_optimizer_matches_brute_force_fractional_cell_integration():
    generator = np.random.default_rng(12)
    for _ in range(40):
        w, h = (int(v) for v in generator.integers(5, 40, 2))
        heatmap = generator.uniform(0, 1, (5, 7))
        ratio = float(generator.uniform(0.2, 4))
        actual = solve_crop(heatmap, w, h, ratio)
        scores = [
            score_crop(heatmap, Crop(x, y, actual.width, actual.height, 0), w, h)
            for x in range(w - actual.width + 1)
            for y in range(h - actual.height + 1)
        ]
        assert actual.retained_importance == pytest.approx(max(scores), abs=1e-10)
        assert actual.retained_importance == pytest.approx(score_crop(heatmap, actual, w, h))


def test_invalid_activations_are_ignored():
    heatmap = np.array([[np.nan, np.inf], [-1, 1]])
    assert focal_point(heatmap) == {"x": 0.75, "y": 0.75}


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_aspect_ratio_fails(ratio):
    with pytest.raises(ValueError, match="Aspect ratio"):
        solve_crop(np.ones((2, 2)), 100, 100, ratio)
