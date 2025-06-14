import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


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
    def __init__(self):
        super().__init__()

        # todo 레이어 깊고 채널 적게, 레이어 얕고 채널 많게
        self.feature_blocks = nn.ModuleList([
            self._make_single_conv_block(3, 60, 3, 2, 1),  # 320x320 -> 160x160
            self._make_single_conv_block_with_db(60, 120, 3, 2, 1, 0.15, 4),  # 160x160 -> 80x80
            # self._make_single_conv_block(24, 48, 3, 2, 1),  # 80x80 -> 40x40
            # self._make_single_conv_block(48, 48, 3, 2, 1),  # 40x40 -> 20x20
            # self._make_single_conv_block(48, 96, 3, 2, 1),  # 20x20 -> 10x10
            # self._make_single_conv_block(96, 96, 3, 2, 1),  # 10x10 -> 5x5
            # self._make_single_conv_block(96, 192, 3, 2, 1),  # 5x5 -> 3x3
            # self._make_single_conv_block(192, 192, 3, 1, 0),  # 3x3 -> 1x1
        ])

        self.reduces = nn.ModuleList()

        # 초기 out_channels: 첫 block 출력 채널 수
        prev_channels = self.feature_blocks[0][0].out_channels

        # reduce layer는 두 번째 block부터 필요하므로 blocks[1:] 기준으로 생성
        for block in self.feature_blocks[1:]:
            block_out_ch = block[0].out_channels
            concat_ch = prev_channels + block_out_ch
            self.reduces.append(
                nn.Sequential(
                    nn.Conv2d(concat_ch, concat_ch // 2, kernel_size=1, bias=False),
                    RMSNorm(concat_ch // 2),
                    nn.SiLU()
                )
            )
            prev_channels = concat_ch // 2

    def _make_single_conv_block(self, in_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            RMSNorm(out_ch),
            nn.SiLU()
        )

    def _make_single_conv_block_with_db(self, in_ch, out_ch, ks, strd, pdd, bp, bs):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            RMSNorm(out_ch),
            nn.SiLU(),
            DropBlock2D(drop_prob=bp, block_size=bs)
        )

    def forward(self, x):
        out = None
        target_h, target_w = None, None

        for i, block in enumerate(self.feature_blocks):
            x = block(x)

            if out is None:
                out = x
                target_h, target_w = x.shape[2], x.shape[3]
            else:
                f = F.interpolate(x, size=(target_h, target_w), mode='nearest')
                out = torch.cat([out, f], dim=1)
                reduce_layer = self.reduces[i - 1]  # reduce_layer는 두 번째 block부터 있음
                out = reduce_layer(out)

        return out


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = UpsampleConcatBackbone()

        dummy_input = torch.zeros(1, 3, 320, 320)  # (B, C, H, W)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = num_classes * 10

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, C, 1, 1)
            nn.Flatten(),  # (B, C)
            nn.LayerNorm(backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(backbone_output_ch, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# import timm  # EfficientNetV2-S 포함 다양한 pretrained 모델 제공
#
#
# class ConvNeXtV2Backbone(nn.Module):
#     def __init__(self, model_name: str = "convnextv2_tiny"):
#         super().__init__()
#         # pretrained ConvNeXtV2, classifier 제외
#         self.model = timm.create_model(model_name, pretrained=False, features_only=True)
#         # features_only=True: FC layer 제외하고 feature map만 반환
#
#     def forward(self, x):
#         features = self.model(x)  # list of feature maps (보통 4~5개 계층 출력됨)
#         return features[-1]  # 마지막 feature map 사용 (B, C, H, W)
#
#
# class UpsampleConcatClassifier(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.backbone = ConvNeXtV2Backbone()
#
#         # output 채널 수 자동 추출
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 3, 320, 320)
#             dummy_out = self.backbone(dummy_input)
#         self.output_channels = dummy_out.shape[1]
#
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(self.output_channels, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x
