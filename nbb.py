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

        # todo : 처음 빼고 다 depthwise 로 해보기
        self.feature_blocks = nn.ModuleList([
            self._make_single_conv_block(1, 40, 20, 3, 3, 0),  # 243x243 -> 81x81
            self._make_single_conv_block(20, 80, 40, 3, 3, 0),  # 81x81 -> 27x27
            self._make_single_conv_block(40, 160, 80, 3, 3, 0),  # 27x27 -> 9x9
            self._make_single_conv_block(80, 320, 160, 3, 3, 0),  # 9x9 -> 3x3
            self._make_single_conv_block(160, 640, 320, 3, 1, 0),  # 3x3 -> 1x1
        ])

    def _make_single_conv_block(self, in_ch, mid_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # 커널로 형태 비교
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            RMSNorm(mid_ch),
            nn.SiLU(),

            # 커널 단위로 추출된 채널 벡터의 조합으로 고유값 반환(차원 밀도 높이기)
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            RMSNorm(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        target_h, target_w = x.shape[2], x.shape[3]

        feat_list = [x]

        x = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]

        for i, block in enumerate(self.feature_blocks):
            x = block(x)

            f = F.interpolate(x, size=(target_h, target_w), mode='nearest')
            feat_list.append(f)

        out = torch.cat(feat_list, dim=1)

        return out


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = UpsampleConcatBackbone()

        dummy_input = torch.zeros(1, 3, 243, 243)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = num_classes * 10

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
