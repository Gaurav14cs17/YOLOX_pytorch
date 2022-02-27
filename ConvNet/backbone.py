from model_utils.model_block import *
from neck import *


class Module_block_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Module_block_1, self).__init__()
        self.conv_block = BaseConv(in_channels, out_channels, ksize=3, stride=2)
        self.mix_block = ConvNextBlock(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.mix_block(x)
        return x


class Module_block_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Module_block_2, self).__init__()
        self.conv_block = BaseConv(in_channels, out_channels, ksize=3, stride=2)
        self.mix_block = mixer(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.mix_block(x)
        return x


class Model(nn.Module):
    def __init__(self, in_channels=3):
        super(Model, self).__init__()
        self.focus_layer = Focus(in_channels, 32, ksize=3)
        self.mb1_layer_1 = Module_block_1(32, 64)
        self.mb1_layer_2 = Module_block_1(64, 64)
        self.mb2_layer_1 = Module_block_2(64, 64)
        self.mb2_layer_2 = Module_block_2(64, 64)
        self.mb2_layer_3 = Module_block_2(64, 64)
        self.last_layer = nn.Sequential(BaseConv(64, 64, 3, 2),
                                        SPPBottleneck(64, 64), mixer(64))

    def forward(self, x):
        x = self.focus_layer(x)
        x = self.mb1_layer_1(x)
        x = self.mb1_layer_2(x)
        c0 = self.mb2_layer_1(x)
        c1 = self.mb2_layer_2(c0)
        c2 = self.mb2_layer_3(c1)
        c3 = self.last_layer(c2)
        return [c0, c1, c2, c3]


if __name__ == '__main__':
    obj = Model()
    image = torch.randn((1, 3, 640, 640))
    obj(image)
    torch.save(obj.state_dict(), 'model_mixer.pth')
