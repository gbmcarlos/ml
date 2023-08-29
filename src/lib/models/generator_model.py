import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)

        )

    def forward(self, x):
        return self.conv(x)


# Starting with x like (N, 3, 256, 256), output a z like (N, 1, 256, 256)
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='reflect')
        ) # (N, in_channels, 256, 256) -> (N, 64, 128, 128)

        self.encoder_block_1 = EncoderBlock(64, 128) # (N, 64, 128, 128)-> (N, 128, 64, 64)
        self.encoder_block_2 = EncoderBlock(128, 256) # (N, 128, 64, 64) -> (N, 256, 32, 32)
        self.encoder_block_3 = EncoderBlock(256, 512) # (N, 256, 32, 32) -> (N, 512, 16, 16)
        self.encoder_block_4 = EncoderBlock(512, 1024) # (N, 512, 16, 16) -> (N, 1024, 8, 8)
        self.encoder_block_5 = EncoderBlock(1024, 1024) # (N, 1024, 8, 8) -> (N, 1024, 4, 4)

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU()
        ) # (N, 1024, 4, 4) -> (N, 1024, 4, 4)

        self.decoder_block_1 = DecoderBlock(1024, 1024) # (N, 1024, 4, 4) -> (N, 1024, 8, 8)
        self.decoder_block_2 = DecoderBlock(1024*2, 512) # (N, 1024*2, 8, 8) -> (N, 512, 16, 16)
        self.decoder_block_3 = DecoderBlock(512*2, 256) # (N, 512*2, 16, 16) -> (N, 256, 32, 32)
        self.decoder_block_4 = DecoderBlock(256*2, 128) # (N, 256*2, 32, 32) -> (N, 128, 64, 64)
        self.decoder_block_5 = DecoderBlock(128*2, 64) # (N, 128*2, 32, 32) -> (N, 64, 128, 128)

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ) # (N, 64*2, 128, 128) -> (N, out_channels, 256, 256)

    def forward(self, x):

        encoded_0 = self.initial_block(x)
        encoded_1 = self.encoder_block_1(encoded_0)
        encoded_2 = self.encoder_block_2(encoded_1)
        encoded_3 = self.encoder_block_3(encoded_2)
        encoded_4 = self.encoder_block_4(encoded_3)

        bottleneck = self.bottleneck_block(encoded_4)

        decoded_1 = self.decoder_block_1(bottleneck)
        decoded_2 = self.decoder_block_2(torch.cat([decoded_1, encoded_4], dim=1))
        decoded_3 = self.decoder_block_3(torch.cat([decoded_2, encoded_3], dim=1))
        decoded_4 = self.decoder_block_4(torch.cat([decoded_3, encoded_2], dim=1))
        decoded_5 = self.decoder_block_5(torch.cat([decoded_4, encoded_1], dim=1))

        output = self.final_block(torch.cat([decoded_5, encoded_0], dim=1))

        activations = [encoded_0.detach(), encoded_1.detach(), encoded_2.detach(), encoded_3.detach(), encoded_4.detach(), bottleneck.detach(), decoded_1.detach(), decoded_2.detach(), decoded_3.detach(), decoded_4.detach(), decoded_5.detach()]
        
        return output, activations

def test():
    x = torch.randn((1, 3, 256, 256))
    print('x:', x.shape)
    model = Generator()
    pred = model(x)
    print('output:', pred.shape)

if __name__ == "__main__":
    test()