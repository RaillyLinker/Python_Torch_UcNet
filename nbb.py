import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))  # (1, C, 1, 1)로 broadcasting
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()  # 채널 방향 평균
        return (x / rms) * self.weight


class UpsampleConcatBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, concat_stride: int = 2):
        super().__init__()
        self.concat_stride = concat_stride

        assert self.concat_stride > 0, "concat_stride must be positive"

        # 각 블록 채널 수를 원하는 대로 조절 가능
        self.blocks = nn.ModuleList([
            self._make_block(in_channels, 32, 32),  # 512 -> 256
            self._make_block(32, 64, 32),  # 256 -> 128
            self._make_block(32, 64, 64),  # 128 -> 64
            self._make_block(64, 128, 64),  # 64 -> 32
            self._make_block(64, 128, 128),  # 32 -> 16
            self._make_block(128, 256, 128),  # 16 -> 8
            self._make_block(128, 256, 256),  # 8 -> 4
            self._make_block(256, 512, 256),  # 4 -> 2
            self._make_block(256, 512, 512),  # 2 -> 1
        ])

        assert len(self.blocks) >= self.concat_stride, "concat_stride is too large for number of blocks"

        selected_indices = list(range(len(self.blocks) - 1, -1, -concat_stride))
        if 0 not in selected_indices:
            selected_indices.append(0)
        self.selected_indices = selected_indices

        # concat 후 채널 수는 upsampled feature들의 채널 수 합입니다.
        concat_channels = sum(
            self.blocks[i][3].out_channels for i in self.selected_indices
        )
        # concat_channels를 반으로 줄이는 레이어
        self.reduce = nn.Sequential(
            nn.Conv2d(concat_channels, concat_channels // 2, kernel_size=1, bias=False),
            RMSNorm(concat_channels // 2),
            nn.GELU()
        )

    def _make_block(self, in_ch: int, mid_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
            RMSNorm(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            RMSNorm(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)

        target_h, target_w = feats[0].shape[2], feats[0].shape[3]

        upsampled = []
        for i in self.selected_indices:
            f = feats[i]
            if f.shape[2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w), mode='nearest')
            upsampled.append(f)

        out = torch.cat(upsampled, dim=1)
        out = self.reduce(out)  # concat 후 채널 수 절반으로 축소

        return out


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, concat_stride: int = 2):
        super().__init__()
        self.backbone = UpsampleConcatBackbone(in_channels=in_channels, concat_stride=concat_stride)

        # 백본의 reduced 채널 수를 그대로 사용
        self.output_channels = self.backbone.reduce[0].out_channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.output_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x
