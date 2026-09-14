"""Human-preference crop ranking on top of the distilled importance network."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from focalnet.imaging import INPUT_SIZE, MAP_SIZE
from focalnet.model import FocalNet, ModelConfig, load_checkpoint


@dataclass(frozen=True)
class RankConfig:
    feature_channels: int = 16
    hidden_channels: int = 64

    def __post_init__(self):
        if min(self.feature_channels, self.hidden_channels) <= 0:
            raise ValueError("Ranking feature and hidden channels must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class HumanCropNet(nn.Module):
    """Score arbitrary normalized source-image crop boxes in one shared forward pass."""

    def __init__(self, base: FocalNet, config: RankConfig | None = None):
        super().__init__()
        self.base = base
        self.config = config or RankConfig()
        self.rank_projection = nn.Sequential(
            nn.Conv2d(base.config.decoder_channels, self.config.feature_channels, 1, bias=False),
            nn.BatchNorm2d(self.config.feature_channels),
            nn.ReLU(inplace=True),
        )
        descriptor_channels = self.config.feature_channels + 1
        geometry_features = 11
        self.rank_head = nn.Sequential(
            nn.LayerNorm(descriptor_channels * 3 + geometry_features),
            nn.Linear(descriptor_channels * 3 + geometry_features, self.config.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_channels, self.config.hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_channels // 2, 1),
        )
        centers = (torch.arange(INPUT_SIZE // 8, dtype=torch.float32) + 0.5) / (INPUT_SIZE // 8)
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        self.register_buffer("grid_x", xx.flatten()[None, None], persistent=False)
        self.register_buffer("grid_y", yy.flatten()[None, None], persistent=False)

    def _region_masks(
        self, boxes: torch.Tensor, content: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, top, right, bottom = content.unbind(-1)
        content_width = (right - left).clamp_min(1e-6)
        content_height = (bottom - top).clamp_min(1e-6)
        sample_boxes = torch.stack(
            (
                left[:, None] + boxes[..., 0] * content_width[:, None],
                top[:, None] + boxes[..., 1] * content_height[:, None],
                left[:, None] + boxes[..., 2] * content_width[:, None],
                top[:, None] + boxes[..., 3] * content_height[:, None],
            ),
            dim=-1,
        )
        inside = (
            (self.grid_x >= sample_boxes[..., 0:1])
            & (self.grid_x <= sample_boxes[..., 2:3])
            & (self.grid_y >= sample_boxes[..., 1:2])
            & (self.grid_y <= sample_boxes[..., 3:4])
        ).to(dtype)
        content_mask = (
            (self.grid_x >= left[:, None, None])
            & (self.grid_x <= right[:, None, None])
            & (self.grid_y >= top[:, None, None])
            & (self.grid_y <= bottom[:, None, None])
        ).to(dtype)
        outside = (content_mask - inside).clamp_min(0)
        return inside, outside, content_mask

    def _pool_regions(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        content: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average features inside and outside each crop, excluding letterbox padding."""
        inside, outside, _ = self._region_masks(boxes, content, features.dtype)
        flattened = features.flatten(2).transpose(1, 2)
        inside_features = torch.matmul(inside, flattened) / inside.sum(-1, keepdim=True).clamp_min(
            1
        )
        outside_features = torch.matmul(outside, flattened) / outside.sum(
            -1, keepdim=True
        ).clamp_min(1)
        return inside_features, outside_features

    def importance_retention(
        self, importance_logits: torch.Tensor, boxes: torch.Tensor, content: torch.Tensor
    ) -> torch.Tensor:
        """Approximate retained importance for candidate ranking baselines and diagnostics."""
        importance = F.interpolate(
            importance_logits.sigmoid(),
            size=(INPUT_SIZE // 8, INPUT_SIZE // 8),
            mode="bilinear",
            align_corners=False,
        )
        inside, _, content_mask = self._region_masks(boxes, content, importance.dtype)
        flattened = importance.flatten(2).transpose(1, 2)
        retained = torch.matmul(inside, flattened).squeeze(-1)
        total = torch.matmul(content_mask, flattened).squeeze(-1).clamp_min(1e-8)
        return retained / total

    def forward(
        self, image: torch.Tensor, boxes: torch.Tensor, content: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = self.base.feature_pyramid(image)
        importance_logits = self.base.importance_logits(combined)
        importance = F.interpolate(
            importance_logits.sigmoid(),
            size=combined.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat((self.rank_projection(combined), importance), dim=1)
        inside, outside = self._pool_regions(features, boxes, content)
        width = (boxes[..., 2] - boxes[..., 0]).clamp_min(1e-6)
        height = (boxes[..., 3] - boxes[..., 1]).clamp_min(1e-6)
        content_width = (content[..., 2] - content[..., 0]).clamp_min(1e-6)
        content_height = (content[..., 3] - content[..., 1]).clamp_min(1e-6)
        geometry = torch.cat(
            (
                boxes,
                width[..., None],
                height[..., None],
                (width * height)[..., None],
                torch.log(width * content_width[:, None] / (height * content_height[:, None]))[
                    ..., None
                ],
                ((boxes[..., 0] + boxes[..., 2]) / 2)[..., None],
                ((boxes[..., 1] + boxes[..., 3]) / 2)[..., None],
                torch.log(content_width / content_height)[:, None, None].expand(
                    -1, boxes.shape[1], -1
                ),
            ),
            dim=-1,
        )
        descriptors = torch.cat((inside, outside, inside - outside, geometry), dim=-1)
        return importance_logits, self.rank_head(descriptors).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def ranking_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.rank_projection, self.rank_head)
            for parameter in module.parameters()
        )


class HumanCropInference(nn.Module):
    def __init__(self, model: HumanCropNet):
        super().__init__()
        self.model = model

    def forward(
        self, image: torch.Tensor, boxes: torch.Tensor, content: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        importance, scores = self.model(image, boxes, content)
        return importance.sigmoid(), scores


def initialize_human_model(
    base_checkpoint: str | Path, config: RankConfig | None = None
) -> tuple[HumanCropNet, dict]:
    base, checkpoint = load_checkpoint(str(base_checkpoint))
    return HumanCropNet(base, config), checkpoint


def load_human_checkpoint(path: str | Path) -> tuple[HumanCropNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or checkpoint.get("task") != "human-crop-ranking":
        raise ValueError("Unsupported human crop checkpoint format")
    if checkpoint.get("input_size") != INPUT_SIZE or checkpoint.get("map_size") != MAP_SIZE:
        raise ValueError("Checkpoint preprocessing dimensions do not match this version")
    base = FocalNet(ModelConfig(**checkpoint["model_config"]))
    model = HumanCropNet(base, RankConfig(**checkpoint["rank_config"]))
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
