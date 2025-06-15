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


class MultiscaleConcatNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 입력 해상도, 구역 분할 수 변화는 3의 배수(3-> 9 -> 27 -> 81 -> 243 -> 729)
        # 구역 분할 수가 적을수록 채널 수를 늘리기

        self.detail_four_feats_conv = self._make_single_conv_block(1, 20, 3, 3, 0)  # 243x243 -> 81x81
        self.detail_three_feats_conv = self._make_single_conv_block(1, 40, 9, 9, 0)  # 243x243 -> 27x27
        self.detail_two_feats_conv = self._make_single_conv_block(1, 80, 27, 27, 0)  # 243x243 -> 9x9
        self.detail_one_feats_conv = self._make_single_conv_block(1, 160, 81, 81, 0)  # 243x243 -> 3x3
        self.global_feats_conv = self._make_single_conv_block(1, 320, 243, 243, 0)  # 243x243 -> 1x1

    def _make_single_conv_block(self, in_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            RMSNorm(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        target_h, target_w = x.shape[2], x.shape[3]

        x_gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        detail_four_feats = self.detail_four_feats_conv(x_gray)
        detail_three_feats = self.detail_three_feats_conv(x_gray)
        detail_two_feats = self.detail_two_feats_conv(x_gray)
        detail_one_feats = self.detail_one_feats_conv(x_gray)
        global_feats = self.global_feats_conv(x_gray)

        out = torch.cat(
            [
                x,
                F.interpolate(detail_four_feats, size=(target_h, target_w), mode='nearest'),
                F.interpolate(detail_three_feats, size=(target_h, target_w), mode='nearest'),
                F.interpolate(detail_two_feats, size=(target_h, target_w), mode='nearest'),
                F.interpolate(detail_one_feats, size=(target_h, target_w), mode='nearest'),
                F.interpolate(global_feats, size=(target_h, target_w), mode='nearest')
            ],
            dim=1
        )

        return out


# ----------------------------------------------------------------------------------------------------------------------
# class UpsampleConcatClassifier(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.backbone = MultiscaleConcatNetwork()
#
#         dummy_input = torch.zeros(1, 3, 243, 243)
#         with torch.no_grad():
#             backbone_output = self.backbone(dummy_input)
#
#         _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape
#
#         self.conv1x1 = nn.Conv2d(
#             in_channels=backbone_output_ch,
#             out_channels=num_classes,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
#
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.backbone(x)
#         x = self.conv1x1(x)
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)
#         return x

class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = MultiscaleConcatNetwork()

        dummy_input = torch.zeros(1, 3, 243, 243)  # (B, C, H, W)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = num_classes * 10

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, C, 1, 1)
            nn.Flatten(),  # (B, C)
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
