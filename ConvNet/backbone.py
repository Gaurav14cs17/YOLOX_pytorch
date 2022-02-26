from model_utils.model_block import *


class Model(nn.Module):
    def __init__(self, in_channels=3, use_block='mixer'):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(3, 3), padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=3)
        Block = ConvNextBlock if (use_block == 'ConNextBlock') else mixer
        self.convNetBlock_1 = Block(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=2)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.bu_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)
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
        x = self.conv12(x)
        x = self.convNetBlock_1(x)
        x = self.conv2(x)
        x = self.conv21(x)
        x = self.convNetBlock_2(x)
        c0 = self.convNetBlock_3(x)

        x = self.conv3(c0)
        c1 = self.convNetBlock_4(x) + self.bu_conv2(c0)

        x = self.conv4(x)
        c2 = self.convNetBlock_5(x) + self.bu_conv2(c1)

        x = self.conv5(c2)
        c3 = self.convNetBlock_6(x) + self.bu_conv2(c2)

        return [c0, c1, c2, c3]


if __name__ == '__main__':
    obj = Model(use_block='mixer')
    image = torch.randn((1, 3, 416, 416))
    obj(image)
    for x in obj(image):
        print(x.shape)
    # torch.save(obj.state_dict(), 'model_mixer.pth')
