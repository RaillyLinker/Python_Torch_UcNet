import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConcatBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # 시각의 중요 요소 : 색상, 점, 명암, 선, 도형, 질감, 공간감, 선명도
        # 색상은 앞에서 이미 가져옴 Clear
        # 점은 생상과 동시에 존재함 Clear

        # 명암은 색상쪽에서 가져오면 될 것 같음

        # 선은 점이 모여서 나오고 선이 모여서도 나옴.(선의 종류 N개를 채널로 선정)
        # 도형은 선이 모여서 나옴(도형의 종류 N개를 채널로 선정)
        # 질감은 종합(단단함 - 부드러움, 건조함 - 축축함, 이런식으로 몇가지를 나누면 될 듯)
        # 선명도는 질감과 비슷해보임
        # 공간감(깊이감으로 할까?)

        # todo : 서로 다른 스케일에서 시각 요소들 추출

        self.feature_blocks = nn.ModuleList([
            self._make_single_conv_block(1, 32, 4, 3, 2, 1),  # 320x320 -> 160x160

            self._make_single_conv_block(4, 128, 64, 3, 2, 1),  # 160x160 -> 80x80
            self._make_single_conv_block(64, 128, 64, 3, 2, 1),  # 80x80 -> 40x40
            self._make_single_conv_block(64, 128, 64, 3, 2, 1),  # 40x40 -> 20x20
        ])

    def _make_single_conv_block(self, in_ch, mid_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),

            # 픽셀당 패턴을 파악 및 의미 추출
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        target_h, target_w = x.shape[2], x.shape[3]

        # 원본 컬러 이미지 저장
        feat_list = [x]

        # 분석을 위한 흑백 변환(컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분)
        x = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]

        for i, block in enumerate(self.feature_blocks):
            # 특징 추출
            x = block(x)
            # 추출된 특징 upSampling 및 특징 리스트에 입력
            feat_list.append(F.interpolate(x, size=(target_h, target_w), mode='nearest'))

        # 특징 리스트들을 전부 concat
        out = torch.cat(feat_list, dim=1)

        return out


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = UpsampleConcatBackbone()

        dummy_input = torch.zeros(2, 3, 320, 320)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = backbone_output_ch * 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(backbone_output_ch, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x
