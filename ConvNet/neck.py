import torch
import torch.nn as nn
from model_utils.model_block import up_sampling


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias, )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels=None, depthwise=True, act="silu"):
        super().__init__()
        if in_channels is None:
            in_channels = [128, 64, 128, 64]
            strides = [64 , 32 , 16 , 8 ]
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        # self.up = up_sampling(64 , 64 , 2 )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_c3 = Conv(in_channels[-1], in_channels[-1], 1, 1, act=act)
        self.csp_c2 = CSPLayer(in_channels[-2], in_channels[-1], n=1, depthwise=depthwise)
        self.conv_c2 = Conv(in_channels[-1], in_channels[-1], 1, 1, act=act)
        self.csp_c1 = CSPLayer(in_channels[-2], in_channels[-1], n=1, depthwise=depthwise)
        self.conv_c1 = Conv(in_channels[-1], in_channels[-1], 1, 1, act=act)
        self.csp_c0 = CSPLayer(in_channels[-2], in_channels[-1], n=1, depthwise=depthwise)
        #################################################################################
        self.c1out = Conv(in_channels[-1], in_channels[-1], 3, 2, act=act)
        self.c1out_csp = CSPLayer(in_channels[-2], in_channels[-1], n=1, depthwise=depthwise)
        self.c2out = Conv(in_channels[-1], in_channels[-1], 3, 2, act=act)
        self.c2out_csp = CSPLayer(in_channels[-2], in_channels[-1], n=1, depthwise=depthwise)
        self.c3out = Conv(in_channels[-1], in_channels[-1], 3, 2, act=act)
        self.c3out_csp = CSPLayer(in_channels[-2], in_channels[-1], n=1, depthwise=depthwise)


    def forward(self, features):
        [x0, x1, x2, x3] = features
        c3_out = self.conv_c3(x3)
        c2_out = torch.cat([x2, self.up(c3_out)], dim=1)
        c2_out = self.csp_c2(c2_out)
        c2_out = self.conv_c2(c2_out)
        c1_out = torch.cat([x1, self.up(c2_out)], dim=1)
        c1_out = self.csp_c1(c1_out)
        c1_out = self.conv_c1(c1_out)
        c0_out = torch.cat([x0, self.up(c1_out)], dim=1)
        c0_out = self.csp_c0(c0_out)
        #################################################################################
        n1_out = self.c1out(c0_out)
        n1_out = torch.cat([n1_out, c1_out], dim=1)
        n1_out = self.c1out_csp(n1_out)
        n2_out = self.c2out(n1_out)
        n2_out = torch.cat([n2_out, c2_out], dim=1)
        n2_out = self.c2out_csp(n2_out)
        n3_out = self.c3out(n2_out)
        n3_out = torch.cat([n3_out, c3_out], dim=1)
        n3_out = self.c3out_csp(n3_out)
        return [c0_out, n1_out, n2_out, n3_out]


if __name__ == '__main__':
    model_obj = FPN()
    image_list = []
    w, h = 40, 40
    for _ in range(4):
        image = torch.randn((1, 64, w, h))
        w = w // 2
        h = h // 2
        image_list.append(image)

    print('start')
    for x in image_list:
        print(x.shape)

    print("--" * 25)
    neck_output = model_obj(image_list)
    for y in neck_output:
        print(y.shape)
