from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numbers import Integral

__all__ = ['ESNet']

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
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, groups=groups, bias=False)
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
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                               padding=0, groups=min(channel, channel // reduction))
        self.conv2 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                               padding=0, groups=min(channel, channel // reduction))

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


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(in_channels=in_channels // 2, out_channels=mid_channels // 2, kernel_size=1,
                                    stride=1, padding=0, groups=4, act=act)
        self._conv_dw = ConvBNLayer(in_channels=mid_channels // 2, out_channels=mid_channels // 2, kernel_size=3,
                                    stride=stride, padding=1, groups=mid_channels // 2, act=None)
        self._se = SEModule(mid_channels)
        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels, out_channels=out_channels // 2, kernel_size=1, stride=1,
            padding=0, groups=4, act=act)

    def forward(self, inputs):
        x1, x2 = torch.chunk(inputs, chunks=2, dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], axis=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], axis=1)
        out = channel_shuffle(out, 2)
        return out


class InvertedResidualDS(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act="relu"):
        super(InvertedResidualDS, self).__init__()
        # branch1
        self._conv_dw_1 = ConvBNLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                                      padding=1,
                                      groups=in_channels, act=None)
        self._conv_linear_1 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1,
                                          stride=1,
                                          padding=0, groups=6, act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(in_channels=in_channels, out_channels=mid_channels // 2, kernel_size=1,
                                      stride=1, padding=0, groups=2, act=act)

        self._conv_dw_2 = ConvBNLayer(in_channels=mid_channels // 2, out_channels=mid_channels // 2, kernel_size=3,
                                      stride=stride, padding=1, groups=mid_channels // 2, act=None)

        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(in_channels=mid_channels // 2, out_channels=out_channels // 2, kernel_size=1,
                                          stride=1, padding=0, groups=4, act=act)
        self._conv_dw_mv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                        stride=1, padding=1, groups=out_channels, act="hard_swish")

        self._conv_pw_mv1 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                                        stride=1, padding=0, groups=1, act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], axis=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out


class ESNet(nn.Module):
    def __init__(self,
                 scale=0.75,
                 act="hard_swish",
                 feature_maps=None,
                 channel_ratio=None):
        super(ESNet, self).__init__()
        if channel_ratio is None:
            channel_ratio = [0.875, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        if feature_maps is None:
            feature_maps = [2, 9, 12]
        self.scale = scale
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]

        stage_out_channels = [
            -1, 24, make_divisible(128 * scale), make_divisible(256 * scale),
            make_divisible(512 * scale), 1024
        ]

        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            groups=3,
            act=act)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottleneck sequences
        block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = InvertedResidualDS(
                        in_channels=stage_out_channels[stage_id + 1],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=2,
                        act=act)
                else:
                    block = InvertedResidual(
                        in_channels=stage_out_channels[stage_id + 2],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=1,
                        act=act)
                block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)

        self._block_list = nn.ModuleList(block_list)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)
        return outs


if __name__ == "__main__":
    from thop import profile

    m = ESNet()
    # m.init_weights()
    m.eval()
    inputs = torch.rand(1, 3, 416, 416)
    total_ops, total_params = profile(m, (inputs,))
    print("total_ops {}G, total_params {}M".format(total_ops / 1e9, total_params / 1e6))
    level_outputs = m(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


    '''
    total_ops 0.152034983G, total_params 0.299095M
    (1, 96, 52, 52)
    (1, 192, 26, 26)
    (1, 384, 13, 13)
    '''
