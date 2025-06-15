import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConcatBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        img_ch = 3

        self.feature_blocks = nn.ModuleList([
            self._make_single_conv_block(1, 512, 8, 3, 2, 1),  # 320x320 -> 160x160
            self._make_single_conv_block(8, 512, 16, 3, 2, 1),  # 160x160 -> 80x80
            self._make_single_conv_block(16, 512, 32, 3, 2, 1),  # 80x80 -> 40x40
            self._make_single_conv_block(32, 512, 64, 3, 2, 1),  # 40x40 -> 20x20
            self._make_single_conv_block(64, 512, 128, 3, 2, 1),  # 20x20 -> 10x10
            self._make_single_conv_block(128, 512, 256, 3, 2, 1),  # 10x10 -> 5x5
            self._make_single_conv_block(256, 512, 512, 3, 2, 1),  # 5x5 -> 3x3
            self._make_single_conv_block(512, 512, 1024, 3, 1, 0),  # 3x3 -> 1x1
        ])

        self.reduce_ch_half_list = nn.ModuleList()
        in_channels = img_ch
        for block in self.feature_blocks:
            out_ch = block[3].out_channels
            in_channels = in_channels + out_ch
            half_ch = in_channels // 2
            self.reduce_ch_half_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, half_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(half_ch),
                    nn.SiLU()
                ),
            )
            in_channels = half_ch

    def _make_single_conv_block(self, in_ch, mid_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),

            # 픽셀당 패턴을 파악
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
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

        dummy_input = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = backbone_output_ch * 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm2d(backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(backbone_output_ch, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x
