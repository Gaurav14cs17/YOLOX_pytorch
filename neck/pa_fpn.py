import torch
import torch.nn as nn
from backbone.draknet import BaseConv, CSPLayer, DWConv


class YOLO_PA_FPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0,in_channels=None, depthwise=False, act="silu"):
        super(YOLO_PA_FPN, self).__init__()

        if in_channels is None:
            in_channels = [256, 512, 1024]

        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            act=act
        )

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

    def forward(self, out_features):
        feat1, feat2, feat3 = out_features

        #   20, 20, 1024 -> 20, 20, 512
        P5 = self.lateral_conv0(feat3)
        #  20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        #  (40, 40, 512) + (40, 40, 512) -> 40, 40, 1024
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #   40, 40, 1024 -> 40, 40, 512
        P5_upsample = self.C3_p4(P5_upsample)

        #   40, 40, 512 -> 40, 40, 256
        P4 = self.reduce_conv1(P5_upsample)
        #   40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        #   (80, 80, 256) + (80, 80, 256) -> 80, 80, 512
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        #   80, 80, 512 -> 80, 80, 256
        P3_out = self.C3_p3(P4_upsample)

        #   80, 80, 256 -> 40, 40, 256
        P3_downsample = self.bu_conv2(P3_out)
        #   (40, 40, 256) + (40, 40, 256) -> 40, 40, 512
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        #   40, 40, 256 -> 40, 40, 512
        P4_out = self.C3_n3(P3_downsample)

        #   40, 40, 512 -> 20, 20, 512
        P4_downsample = self.bu_conv1(P4_out)
        #   (20, 20, 512) + (20, 20, 512) -> 20, 20, 1024
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        #   20, 20, 1024 -> 20, 20, 1024
        P5_out = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)
