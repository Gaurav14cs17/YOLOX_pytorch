from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numbers import Integral
from net.model_lib.csp_darknet import DWConv, BaseConv

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 1, groups=1, act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=groups, bias=False)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Identity() if act is None else acts[act]

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self.act(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = DWConv(channel, channel // reduction, 1, 1)
        self.conv2 = DWConv(channel // reduction, channel, 1, 1)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return inputs * outputs


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


'''
BaseConv : in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
Deptg-wise :   in_channels, out_channels, ksize, stride=1, act="silu"
'''


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(in_channels=in_channels // 2, out_channels=mid_channels // 2, kernel_size=1,stride=1, padding=0, groups=4, act=act)
        self._conv_dw = ConvBNLayer(in_channels=mid_channels // 2, out_channels=mid_channels // 2, kernel_size=3,stride=stride, padding=1, groups=mid_channels // 2, act=None)
        self._se = SEModule(mid_channels)
        self._conv_linear = DWConv(in_channels=mid_channels, out_channels=out_channels // 2, ksize=1, stride=1, act=act)

    def forward(self, inputs):
        x1, x2 = torch.chunk(inputs, chunks=2, dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        out = channel_shuffle(out, 2)
        return out


class InvertedResidualDS(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidualDS, self).__init__()
        self._conv_1 = DWConv(in_channels=in_channels, out_channels=out_channels // 2, ksize=1, stride=1,act=act)
        self._conv_2 = DWConv(in_channels=in_channels, out_channels=mid_channels // 2, ksize=1, stride=1, act=act)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = DWConv(in_channels=mid_channels // 2, out_channels=out_channels // 2, ksize=1, stride=1,act=act)
        self._conv_3 = DWConv(in_channels=out_channels, out_channels=out_channels, ksize=3, stride=1, act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_1(inputs)

        x2 = self._conv_2(inputs)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)

        out = torch.cat([x1, x2], dim=1)
        out = self._conv_3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=(stride, stride),bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=(stride, stride),
                          bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

import time
class SSNet(nn.Module):
    def __init__(self):
        super(SSNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1))
        self.resd_block = BasicBlock(16, 32)
        self.inverted_residual_block = InvertedResidual(32, 64, 64, stride=1)
        #self.inverted_IDs_block = InvertedResidualDS(48, 64, 64, stride=2)

    def forward(self, x):
        t1 = time.time()
        x = self.conv1(x)
        t2 = time.time()
        print("conv1 : ",(t2-t1))
        t1 = t2
        x_resd = self.resd_block(x)
        t2 = time.time()
        print("x_resd : ", (t2 - t1))
        t1 = t2
        x_inverted_block = self.inverted_residual_block(x_resd)
        t2 = time.time()
        print("x_inverted_block : ", (t2 - t1))
        t1 = t2
        # x_inverted_ids = self.inverted_IDs_block(x_inverted_block)
        # t2 = time.time()
        # print("x_inverted_ids : ", (t2 - t1))
        # t1 = t2
        # print(x_inverted_ids.shape)


if __name__ == "__main__":
    from thop import profile


    m = SSNet()
    # m.init_weights()
    m.eval()
    inputs = torch.rand(1, 3, 320, 320)
    # total_ops, total_params = profile(m, (inputs,))
    # print("total_ops {}G, total_params {}M".format(total_ops / 1e9, total_params / 1e6))
    t1 = time.time()
    level_outputs = m(inputs)
    print(time.time() - t1)

    '''
   total_ops 1.925401368G, total_params 0.0254M
    (1, 48, 318, 318)
    (1, 64, 159, 159)
    '''
