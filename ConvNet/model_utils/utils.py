import os
import time
import numpy as np
from PIL import Image
import torch ,math
import torch.nn as nn
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import models, transforms


def align_number(number, N):
    assert type(number) == int
    num_str = str(number)
    assert len(num_str) <= N
    return (N-len(num_str))*'0' + num_str


def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y


def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed


def convert2img(x):
    return Image.fromarray(x*255).convert('L')


def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)
    
    
def cache_model(model, path, multi_gpu):
    if multi_gpu==True:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
        
        
def DS2(x):
    return F.max_pool2d(x, 2)


def DS4(x):
    return F.max_pool2d(x, 4)


def DS8(x):
    return F.max_pool2d(x, 8)


def DS16(x):
    return F.max_pool2d(x, 16)


def US2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear')


def US4(x):
    return F.interpolate(x, scale_factor=4, mode='bilinear')


def US8(x):
    return F.interpolate(x, scale_factor=8, mode='bilinear')


def US16(x):
    return F.interpolate(x, scale_factor=16, mode='bilinear')


def RC(F, A):
    return F * A + F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            for f, g in m.named_children():
                print('initialize: ' + f)
                if isinstance(g, nn.Conv2d):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(g.weight)
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
                elif isinstance(g, nn.Linear):
                    nn.init.kaiming_normal_(g.weight, mode='fan_in', nonlinearity='relu')
                    if g.bias is not None:
                        nn.init.zeros_(g.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.AdaptiveMaxPool2d) or isinstance(m, nn.ModuleList) or isinstance(m, nn.BCELoss):
            a=1
        else:
            m.initialize()





class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # need modify by the batchsize/GPU_number
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('+ str(self.in_features) + ' -> '+ str(self.out_features) + ')'


class ChannelAttention_diag(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=8):
        super(ChannelAttention_diag, self).__init__()
        self.inter_channels = in_channels // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_channels, self.inter_channels, bias=False),
                           nn.ReLU(inplace=True))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_channels, self.inter_channels, bias=False),
                           nn.ReLU(inplace=True))
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        device = torch.device("cuda")
        B, C, H, W = ftr.size()
        M = self.inter_channels
        ftr_avg = self.avg_fc(self.avg_pool(ftr).squeeze())
        ftr_max = self.max_fc(self.max_pool(ftr).squeeze())
        cw = F.sigmoid(ftr_avg + ftr_max).unsqueeze(-1)
        b = torch.unsqueeze(torch.eye(M, device=device), 0).expand(B, M, M).cuda()
        return torch.mul(b,cw)

    def initialize(self):
        weight_init(self)


class Conv3ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
    def initialize(self):
        weight_init(self)


class Conv1ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
    def initialize(self):
        weight_init(self)


class FeatureDecoding(nn.Module):
    def __init__(self, deep_channels, shallow_channels, out_channels, is_upsample):
        super(FeatureDecoding, self).__init__()
        self.is_upsample = is_upsample
        self.feature_fusion = Conv3ReLU(deep_channels+shallow_channels, out_channels)

    def forward(self, deep_ftr, shallow_ftr):
        if self.is_upsample:
            deep_ftr = US2(deep_ftr)
        cat_ftr = torch.cat((deep_ftr, shallow_ftr), dim=1)
        fus_ftr = self.feature_fusion(cat_ftr)
        return fus_ftr

    def initialize(self):
        weight_init(self)

class MultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScale, self).__init__()
        self.conv_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_5_5 = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2),  # 空洞卷积
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2, dilation=1),  # 普通卷积
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_7_7 = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3),  # 空洞卷积
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3, dilation=1),  # 普通卷积
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
    def forward(self, ftr):
        ftr_3_3 = self.conv_3_3(ftr)
        ftr_5_5 = self.conv_5_5(ftr)
        ftr_7_7 = self.conv_7_7(ftr)
        return ftr_3_3, ftr_5_5, ftr_7_7
    def initialize(self):
        weight_init(self)


class SpatialAttention_ksize(nn.Module):
    def __init__(self):
        super(SpatialAttention_ksize, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, dilation = 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=2, dilation = 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=3, dilation = 3, bias=False)
    def forward(self, ftr, ksize):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W]
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True) # [B, 1, H, W]
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1) # [B, 2, H, W]
        if ksize == 3: am = F.sigmoid(self.conv3(ftr_cat)) # [B, 1, H, W]
        if ksize == 5: am = F.sigmoid(self.conv5(ftr_cat)) # [B, 1, H, W]
        if ksize == 7: am = F.sigmoid(self.conv7(ftr_cat)) # [B, 1, H, W]
        return am
    def initialize(self):
        weight_init(self)

class SpatialAttention_multiscale(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention_multiscale, self).__init__()
        self.MS = MultiScale(in_channels, in_channels)
        self.SA = SpatialAttention_ksize()
        self.conv_merge = Conv1ReLU(in_channels * 3, in_channels)
        self.conv = Conv1ReLU(2, 1)

    def forward(self, ftr):

        a_3_3, a_5_5, a_7_7 = self.SA(ftr, ksize = 3), self.SA(ftr, ksize = 5), self.SA(ftr, ksize = 7)
        a_add = a_3_3 + a_5_5 + a_7_7


        f_3_3, f_5_5, f_7_7 = self.MS(ftr)
        a_merge = self.SA(f_3_3, ksize = 3) + self.SA(f_5_5, ksize = 3) + self.SA(f_7_7, ksize = 3)

        a_merge = self.conv(torch.cat((a_merge/3, a_add/3), dim=1))
        return a_merge

    def initialize(self):
        weight_init(self)


class GR(nn.Module):
    def __init__(self, in_channels, node_n, squeeze_ratio=8):
        super(GR, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.conv_k = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                      nn.ReLU(inplace=True))
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.ca_diag = ChannelAttention_diag(in_channels)
        self.GCN = GraphConvolution(in_channels, in_channels, bias=False)
        self.delta = nn.Parameter(torch.Tensor([0]))

    def forward(self, ftr):
        device = torch.device("cuda")
        B, C, H, W = ftr.size()
        HW = H * W
        M = C // 8
        b = torch.unsqueeze(torch.eye(HW, device=device), 0).expand(B, HW, HW).cuda()
        One = torch.ones(HW, 1, dtype=torch.float32, device=device).expand(B, HW, 1).cuda() # [B, HW, 1]
        diag = self.ca_diag(ftr) # [B, M, M]

        ftr_k = self.conv_k(ftr).view(B, -1, HW)  # [B, M, HW]
        ftr_q = ftr_k.permute(0, 2, 1)  # [B, HW, M]


        D = torch.bmm(ftr_q, diag)
        D = F.sigmoid(torch.bmm(D, ftr_k))
        D = torch.bmm(D, One)
        D = D ** (-1 / 2)
        D = torch.mul(b, D)

        P = torch.bmm(D, ftr_q)
        Pt = P.permute(0, 2, 1)

        X = ftr.view(B, -1, HW).permute(0, 2, 1)
        LX = torch.bmm(Pt, X)
        LX = torch.bmm(diag, LX)
        LX = torch.bmm(P, LX)
        LX = X - LX

        Y = (X + self.GCN(LX)).permute(0, 2, 1)

        Y = Y.view(B, C, H, W)
        return Y

    def initialize(self):
        weight_init(self)


class GR_channel(nn.Module):
    def __init__(self, in_channels, node_n, squeeze_ratio=8):
        super(GR_channel, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.conv_k = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                      nn.ReLU(inplace=True))
        self.ca_diag = ChannelAttention_diag(in_channels, squeeze_ratio=squeeze_ratio)
        self.GCN = GraphConvolution(in_channels, in_channels, bias=False)
        self.delta = nn.Parameter(torch.Tensor([0]))

    def forward(self, ftr):
        device = torch.device("cuda")
        B0, C0, H0, W0 = ftr.size()
        HW0 = H0 * W0
        ftr = ftr.view(B0, C0, HW0).permute(0, 2, 1)
        ftr = ftr.view(B0, HW0, int(C0/16), 16)
        B, C, H, W = ftr.size()
        HW = H * W
        M = C // 8
        b = torch.unsqueeze(torch.eye(HW, device=device), 0).expand(B, HW, HW).cuda()
        One = torch.ones(HW, 1, dtype=torch.float32, device=device).expand(B, HW, 1).cuda() # [B, HW, 1]
        diag = self.ca_diag(ftr)
        ftr_k = self.conv_k(ftr).view(B, -1, HW)
        ftr_q = ftr_k.permute(0, 2, 1)
        D = torch.bmm(ftr_q, diag)
        D = F.sigmoid(torch.bmm(D, ftr_k))
        D = torch.bmm(D, One)
        D = D ** (-1 / 2)
        D = torch.mul(b, D)

        P = torch.bmm(D, ftr_q)
        Pt = P.permute(0, 2, 1)

        X = ftr.view(B, -1, HW).permute(0, 2, 1)
        LX = torch.bmm(Pt, X)
        LX = torch.bmm(diag, LX)
        LX = torch.bmm(P, LX)
        LX = X - LX
        Y = (X + self.GCN(LX)).view(B0, C0, H0, W0)
        return Y

    def initialize(self):
        weight_init(self)

