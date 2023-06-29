import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=4, bias=False, padding='valid'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=bias, padding=padding)
        self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        # out = self.batch(out)
        out = self.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(channels, out_channels=64, stride=2, kernel_size=4, padding=1, padding_mode='reflect')
        self.relu_1 = nn.ReLU()

        self.cnn_block_1 = CNNBlock(in_channels=64, out_channels=64)
        self.cnn_block_2 = CNNBlock(in_channels=64, out_channels=128)
        self.cnn_block_3 = CNNBlock(in_channels=128, out_channels=256)
        self.cnn_block_4 = CNNBlock(in_channels=256, out_channels=512, stride=1, padding='valid')

    def forward(self, x):
        # out = torch.cat([x, y], dim=1)
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.cnn_block_1(out)
        out = self.cnn_block_2(out)
        out = self.cnn_block_3(out)
        # out = self.cnn_block_4(out)
        return out
