import numpy as np
import pytest
import torch
from PIL import Image

from focalnet.exporting import export_model
from focalnet.imaging import prepare_image, restore_map
from focalnet.model import BACKBONES, FocalNet, ModelConfig
from focalnet.runtime import Predictor
from focalnet.training import importance_loss


@pytest.mark.parametrize("backbone", BACKBONES)
def test_both_encoders_receive_gradients_from_the_importance_head(backbone):
    torch.set_num_threads(1)
    model = FocalNet(ModelConfig(backbone))
    output = model(torch.randn(2, 3, 256, 256))
    assert output.shape == (2, 1, 64, 64)
    assert model.parameter_count() < 6_000_000
    target = torch.zeros_like(output)
    target[:, :, 8:20, 20:40] = 1
    loss, _ = importance_loss(output, target, torch.ones_like(output))
    loss.backward()
    for module in (model.encoder, model.lateral, model.refine):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters())
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


@pytest.mark.integration
@pytest.mark.parametrize("backbone", BACKBONES)
def test_fused_onnx_matches_pytorch_on_oriented_image_maps(tmp_path, backbone):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = FocalNet(ModelConfig(backbone)).eval()
    checkpoint = tmp_path / "test.pt"
    torch.save(
        {
            "format_version": 1,
            "input_size": 256,
            "map_size": 64,
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
            "epoch": -1,
        },
        checkpoint,
    )
    artifact = tmp_path / "focalnet.onnx"
    report = export_model(checkpoint, artifact)
    assert report["max_abs_error"] < 1e-5
    predictor = Predictor(artifact)
    for dimensions in ((201, 303), (511, 123)):
        image = Image.new("RGB", dimensions, (220, 140, 20))
        tensor, box = prepare_image(image)
        with torch.inference_mode():
            reference = restore_map(model(torch.from_numpy(tensor)).sigmoid().numpy()[0, 0], box)
        np.testing.assert_allclose(predictor.heatmap(image), reference, atol=1e-5, rtol=1e-4)
