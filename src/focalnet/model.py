"""A pretrained mobile CNN with a small, additive feature-pyramid decoder."""

from dataclasses import asdict, dataclass

import timm
import torch
from torch import nn
from torch.nn import functional as F

from focalnet.imaging import INPUT_SIZE, MAP_SIZE

# M0.6 is in the authors' repo, but is not registered in the pinned timm release.
BACKBONES = {
    "repvit_m0_9": (1, 2, 3),
    "mobilenetv4_conv_small": (2, 3, 4),
}


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "repvit_m0_9"
    decoder_channels: int = 48

    def __post_init__(self):
        if self.backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {self.backbone}; choose {list(BACKBONES)}")
        if self.decoder_channels <= 0:
            raise ValueError("Decoder channels must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class FocalNet(nn.Module):
    """Forward returns logits; exported inference applies sigmoid once."""

    def __init__(self, config: ModelConfig | None = None, *, pretrained: bool = False):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = timm.create_model(
            self.config.backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=BACKBONES[self.config.backbone],
        )
        if self.encoder.feature_info.reduction() != [8, 16, 32]:
            raise ValueError("Backbone must expose feature strides 8, 16, and 32")
        channels = self.config.decoder_channels
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, channels, 1, bias=False) for c in self.encoder.feature_info.channels()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

    def feature_pyramid(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        combined = self.lateral[0](features[0])
        for lateral, feature in zip(self.lateral[1:], features[1:], strict=True):
            combined = combined + F.interpolate(
                lateral(feature), size=features[0].shape[-2:], mode="bilinear", align_corners=False
            )
        return combined

    def importance_logits(self, combined: torch.Tensor) -> torch.Tensor:
        combined = F.interpolate(
            combined, size=(MAP_SIZE, MAP_SIZE), mode="bilinear", align_corners=False
        )
        return self.refine(combined)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.importance_logits(self.feature_pyramid(image))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class InferenceModel(nn.Module):
    def __init__(self, model: FocalNet):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).sigmoid()


def load_checkpoint(path: str) -> tuple[FocalNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("input_size") != INPUT_SIZE or checkpoint.get("map_size") != MAP_SIZE:
        raise ValueError("Checkpoint preprocessing dimensions do not match this version")
    model = FocalNet(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
