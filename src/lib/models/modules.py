import torch
import torch.nn as nn


class ConvBlock(nn.Module):  # Block of layers that applies 2 convolutions, each followed by a ReLU
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, bias=True, padding='same'):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=bias, padding=padding)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, bias=bias, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):  # Block of layers for each step of a UNet's encoder: double convolution and maxpooling, to extract increasing features
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = ConvBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        convolved = self.conv_block(x)
        encoded = self.max_pool(convolved)
        return convolved, encoded


class DecoderBlock(
    nn.Module):  # Block of layers for each step of a UNet's decoder: transposed convolution, skip connection from the encoder, and 2 double convolutions
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, connection):
        out = self.upconv(x)
        out = torch.cat([connection, out], dim=1)
        out = self.conv_block_1(out)
        out = self.conv_block_2(out)
        return out