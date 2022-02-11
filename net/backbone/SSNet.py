from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numbers import Integral

__all__ = ['SSNet']

acts = {"relu": nn.ReLU(inplace=True),
        "hard_swish": nn.Hardswish()}


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride,padding=padding, bias=False)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Identity() if act is None else acts[act]

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self.act(y)
        return y




class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=(1 , 1)),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=(1 , 1)),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        b = F.interpolate(b, scale_factor=1.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)

class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(in_channels=in_channels//2, out_channels=mid_channels // 2, kernel_size=1,stride=1, padding=0, act=act)
        self._conv_dw = ConvBNLayer(in_channels=mid_channels // 2, out_channels=mid_channels // 2, kernel_size=3,stride=stride, padding=1, act=None)
        self._se = BlockTypeA(mid_channels//2 , mid_channels//2 , mid_channels//2 , mid_channels//2 )
        self._conv_linear = ConvBNLayer(in_channels=mid_channels, out_channels=out_channels // 2, kernel_size=1, stride=1,padding=0, act=act)

    def forward(self, inputs):
        x1, x2 = torch.chunk(inputs, chunks=2, dim=1)
        print()
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = self._se(x2, x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], axis=1)
        out = channel_shuffle(out, 2)
        print("INver-->" , out.shape)
        return out







def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v









class SS_FNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SS_FNet, self).__init__()
        block = InvertedResidual
        input_channel = 32
        width_mult = 1.0
        round_nearest = 8

        inverted_residual_setting = [
            #c, n, s
            [16, 1, 1],
            [32, 1, 2],
           # [2, 3, 2],
           # [64, 4, 2],
        ]
        input_channel = 16
        features = [ConvBNLayer(3, input_channel , kernel_size=3 ,stride=1 , padding=1 , act = "relu" )]

        for c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel*2 ,output_channel, stride ))
                input_channel = output_channel*2
        self.features = nn.Sequential(*features)

        self.fpn_selected = [3, 6, 10]



    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break

            print(i , x.shape)
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)

        c2, c3, c4 = fpn_features
        return c2, c3, c4


    def forward(self, x):
        return self._forward_impl(x)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)









if __name__ == '__main__':
    #obj  = InvertedResidual(64 , 128 , 256 , 2 )
    obj = SS_FNet()
    print(obj)
    image = torch.randn(1, 3 , 320 , 320  )
    out = obj(image)
    print(out.shape)