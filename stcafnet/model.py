from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        maximum = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * self.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.cat(
            [torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)],
            dim=1,
        )
        return x * self.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(7)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.channel(x))


class VisualEncoder(nn.Module):
    def __init__(
        self, feature_dim: int = 512, reduction: int = 16, pretrained: bool = True
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        network = efficientnet_b0(weights=weights)
        self.backbone = network.features
        self.cbam = CBAM(1280, reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(1280, feature_dim)

    def forward(self, image: Tensor) -> Tensor:
        x = self.backbone(image)
        x = self.cbam(x)
        return self.projection(self.pool(x).flatten(1))


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class OlfactoryEncoder(nn.Module):
    def __init__(
        self,
        num_sensors: int = 10,
        channels: tuple[int, int] = (64, 128),
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock1D(num_sensors, channels[0]),
            ConvBlock1D(channels[0], channels[1]),
        )
        self.lstm = nn.LSTM(
            input_size=channels[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size * 2, feature_dim)

    def forward(self, enose: Tensor) -> Tensor:
        if enose.ndim != 3:
            raise ValueError("E-nose input must have shape [batch, time, sensors].")
        x = self.cnn(enose.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.projection(sequence[:, -1, :])


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention over one global token per modality."""

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.vision_queries_enose = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.enose_queries_vision = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.vision_norm = nn.LayerNorm(dim)
        self.enose_norm = nn.LayerNorm(dim)

    def forward(self, vision: Tensor, enose: Tensor) -> tuple[Tensor, Tensor]:
        v = vision.unsqueeze(1)
        e = enose.unsqueeze(1)
        v_context, _ = self.vision_queries_enose(v, e, e, need_weights=False)
        e_context, _ = self.enose_queries_vision(e, v, v, need_weights=False)
        return (
            self.vision_norm(v + v_context).squeeze(1),
            self.enose_norm(e + e_context).squeeze(1),
        )


class VectorGatedFusion(nn.Module):
    def __init__(self, dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, vision: Tensor, enose: Tensor) -> tuple[Tensor, Tensor]:
        joined = torch.cat([vision, enose], dim=-1)
        alpha = torch.sigmoid(self.gate(joined))
        gated = alpha * vision + (1.0 - alpha) * enose
        return self.norm(gated + self.ffn(joined)), alpha


class MultiTaskHead(nn.Module):
    def __init__(self, dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.tasks = nn.ModuleDict(
            {name: nn.Linear(128, 1) for name in ("tvc", "tvbn", "tbars")}
        )

    def forward(self, x: Tensor) -> Tensor:
        shared = self.shared(x)
        return torch.cat([self.tasks[name](shared) for name in self.tasks], dim=1)


@dataclass
class STCAFNetOutput:
    predictions: Tensor
    gate: Tensor


class STCAFNet(nn.Module):
    def __init__(
        self,
        num_sensors: int = 10,
        feature_dim: int = 512,
        cbam_reduction: int = 16,
        enose_channels: tuple[int, int] = (64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_heads: int = 8,
        fusion_dropout: float = 0.1,
        head_dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.visual = VisualEncoder(
            feature_dim, cbam_reduction, pretrained=pretrained
        )
        self.olfactory = OlfactoryEncoder(
            num_sensors,
            enose_channels,
            lstm_hidden,
            lstm_layers,
            lstm_dropout,
            feature_dim,
        )
        self.cross_attention = CrossModalAttention(feature_dim, attention_heads)
        self.fusion = VectorGatedFusion(feature_dim, fusion_dropout)
        self.head = MultiTaskHead(feature_dim, head_dropout)
        self.log_variances = nn.Parameter(torch.zeros(3))
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        for name, parameter in self.named_parameters():
            if name.startswith("visual.backbone"):
                continue
            if name == "log_variances":
                nn.init.zeros_(parameter)
            elif parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)

    def freeze_visual_backbone(self, frozen: bool = True) -> None:
        for parameter in self.visual.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(self, image: Tensor, enose: Tensor) -> STCAFNetOutput:
        vision = self.visual(image)
        odor = self.olfactory(enose)
        vision, odor = self.cross_attention(vision, odor)
        fused, gate = self.fusion(vision, odor)
        return STCAFNetOutput(self.head(fused), gate)

    def uncertainty_weighted_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        task_mse = torch.mean((predictions - targets) ** 2, dim=0)
        return torch.sum(0.5 * torch.exp(-self.log_variances) * task_mse + 0.5 * self.log_variances)
