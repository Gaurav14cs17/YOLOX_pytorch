import torch
import torch.nn as nn
import torch.nn.functional as F
#
#
# class DWConvblock(nn.Module):
#     def __init__(self, input_channels, output_channels, size):
#         super(DWConvblock, self).__init__()
#         self.size = size
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#
#         self.block = nn.Sequential(
#             nn.Conv2d(output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False),
#             nn.BatchNorm2d(output_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(output_channels),
#
#             nn.Conv2d(output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False),
#             nn.BatchNorm2d(output_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(output_channels),
#         )
#
#     def forward(self, x):
#         x = self.block(x)
#         return x


class LightFPN(nn.Module):
    def __init__(self, input1_depth , input2_depth, input3_depth, out_depth):
        super(LightFPN, self).__init__()

        self.conv1x1_1 = nn.Sequential(nn.Conv2d(input1_depth, out_depth, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

        self.conv1x1_2 = nn.Sequential(nn.Conv2d(input2_depth, out_depth, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

        self.conv1x1_3 = nn.Sequential(nn.Conv2d(input3_depth, out_depth, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

    def forward(self, C1 , C2, C3):
        S3 = self.conv1x1_3(C3)

        P2 = F.interpolate(C3, scale_factor=2)
        P2 = torch.cat((P2, C2), 1)
        S2 = self.conv1x1_2(P2)

        P1 = F.interpolate(C2 , scale_factor=2)
        P1 = torch.cat((P1 ,C1 ) , 1 )
        S1 = self.conv1x1_1(P1)
        return S2, S3 , S1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
