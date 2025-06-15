import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class UpsampleConcatBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        img_ch = 3

        self.feature_blocks = nn.ModuleList([
            self._make_single_conv_block(1, 512, 16, 3, 2, 1),  # 224x224 -> 112x112
            self._make_single_conv_block(16, 512, 32, 3, 2, 1),  # 112x112 -> 56x56
            self._make_single_conv_block(32, 512, 64, 3, 2, 1),  # 56x56 -> 28x28
            self._make_single_conv_block(64, 512, 128, 3, 2, 1),  # 28x28 -> 14x14
            self._make_single_conv_block(128, 512, 256, 3, 2, 1),  # 14x14 -> 7x7
        ])

        self.reduce_ch_half_list = nn.ModuleList()
        in_channels = img_ch
        for block in self.feature_blocks:
            out_ch = block[3].out_channels
            in_channels = in_channels + out_ch
            half_ch = in_channels // 2
            self.reduce_ch_half_list.append(
                nn.Conv2d(in_channels, half_ch, kernel_size=1, stride=1, padding=0, bias=False)
            )
            in_channels = half_ch

    def _make_single_conv_block(self, in_ch, mid_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            RMSNorm(mid_ch),
            nn.SiLU(),

            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            RMSNorm(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        target_h, target_w = x.shape[2], x.shape[3]

        out = x

        x = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]

        for i, block in enumerate(self.feature_blocks):
            x = block(x)
            out = torch.cat([out, F.interpolate(x, size=(target_h, target_w), mode='nearest')], dim=1)
            out = self.reduce_ch_half_list[i](out)

        return out


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = UpsampleConcatBackbone()

        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = backbone_output_ch * 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.RMSNorm(backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(backbone_output_ch, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x
