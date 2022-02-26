import torch
from torch import nn, einsum
from torch.autograd import grad
from torch.optim import Adam
import torch.nn.functional as F
import torchvision, math
from model_utils.switchable_norm import SwitchNorm2d
from torch.nn.parameter import Parameter
from model_utils.utils import *


class mixer(nn.Module):
    def __init__(self, dim):
        super(mixer, self).__init__()
        self.depthconv = nn.Conv2d(dim, dim, kernel_size=(9, 9), padding=4, groups=dim)
        self.gn1 = SwitchNorm2d(dim)
        self.pointconv = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        self.gn2 = SwitchNorm2d(dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.depthconv(x)
        x = self.gn1(x)
        x = self.gelu(x)
        x = x + shortcut
        x = self.pointconv(x)
        x = self.gn2(x)
        x = self.gelu(x)
        return x


class up_sampling(nn.Module):
    def __init__(self, in_ch, out_ch, stride=8):
        super(up_sampling, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1)), SwitchNorm2d(out_ch), nn.GELU(), )
        dim = out_ch
        self.patchup = nn.ConvTranspose2d(dim, dim, kernel_size=(stride, stride), stride=(stride, stride))
        self.bn2 = SwitchNorm2d(dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.patchup(x)
        x = self.bn2(x)
        output = self.gelu(x)
        return output


class LayerNormChan(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ConvNextBlock(nn.Module):
    def __init__(self, dim, act=nn.ReLU(), ds_kernel_size=7, mult=0.5, ):
        """
        https://arxiv.org/abs/2201.03545
        """
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, (ds_kernel_size, ds_kernel_size), padding=ds_kernel_size // 2, groups=dim),
            LayerNormChan(dim),
            nn.Conv2d(dim, inner_dim, (3, 3), padding=1),
            act,
            nn.Conv2d(inner_dim, dim, (3, 3), padding=1)
        )

    def forward(self, x):
        return self.net(x) + x


class Model(nn.Module):
    def __init__(self, in_channels=3, use_block='ConNextBlock'):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)

        Block = ConvNextBlock if (use_block == 'ConNextBlock') else mixer
        self.convNetBlock_1 = Block(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bu_conv2 = nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=(3,3), stride=(2,2), padding=1)
        self.convNetBlock_2 = Block(64)
        self.convNetBlock_3 = Block(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convNetBlock_4 = Block(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convNetBlock_5 = Block(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.convNetBlock_6 = Block(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.convNetBlock_1(x)

        x = self.conv2(x)
        x = self.convNetBlock_2(x)
        c0 = self.convNetBlock_3(x)

        x = self.conv3(c0)
        c1 = self.convNetBlock_4(x)+ self.bu_conv2(c0)

        x = self.conv4(x)
        c2 = self.convNetBlock_5(x) + self.bu_conv2(c1)

        x = self.conv5(c2)
        c3 = self.convNetBlock_6(x) + self.bu_conv2(c2)

        return [c0, c1, c2, c3]


if __name__ == '__main__':
    obj = Model(use_block='mixer')
    image = torch.randn((1, 3, 320, 320))
    for x in obj(image):
        print(x.shape)
    #torch.save(obj.state_dict(), 'model_mixer.pth')
