import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=True),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True),
			# nn.MaxPool2d(kernel_size=2, stride=2)
		)

	def forward(self, x):
		return self.conv(x)

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, dropout=0.0):
		super().__init__()

		self.conv = nn.Sequential(
			# nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ConvTranspose2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(out_channels),
			nn.Dropout(dropout),
			nn.ReLU(True)
		)

	def forward(self, x, y):
		return self.conv(torch.cat([x, y], dim=1))


# Starting with x like (N, 3, 256, 256), output a z like (N, 1, 256, 256)
class Generator(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.initial_block = EncoderBlock(in_channels, 64) # initial: (N, in_channels, 256, 256) -> (N, 64, 128, 128)
		self.encoder_block_1 = EncoderBlock(64, 128) # encoded_1: (N, 64, 128, 128)-> (N, 128, 64, 64)
		self.encoder_block_2 = EncoderBlock(128, 256) # encoded_2: (N, 128, 64, 64) -> (N, 256, 32, 32)
		self.encoder_block_3 = EncoderBlock(256, 512) # encoded_3: (N, 256, 32, 32) -> (N, 512, 16, 16)
		self.encoder_block_4 = EncoderBlock(512, 1024) # encoded_4: (N, 512, 16, 16) -> (N, 1024, 8, 8)

		self.bottleneck_block = nn.Sequential(
			nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding='same', padding_mode='reflect'),
			nn.ReLU(True)
		) # bottleneck: (N, 1024, 8, 8) -> (N, 1024, 8, 8)

		self.decoder_block_1 = DecoderBlock(1024, 512, 0.5) # decoded_1: (N, 1024*2, 8, 8) -> (N, 512, 16, 16) (bottleneck+encoded_4)
		self.decoder_block_2 = DecoderBlock(512, 256, 0.5) # decoded_2: (N, 512*2, 16, 16) -> (N, 256, 32, 32) (decoded_1+encoded_3)
		self.decoder_block_3 = DecoderBlock(256, 128) # decoded_3: (N, 256*2, 32, 32) -> (N, 128, 64, 64) (decoded_2+encoded_2)
		self.decoder_block_4 = DecoderBlock(128, 64) # decoded_4: (N, 128*2, 64, 64) -> (N, 64, 128, 128) (decoded_3+encoded_1)

		self.final_block = nn.Sequential(
			# nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ConvTranspose2d(in_channels=64*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
			nn.Tanh()
		) # (N, 64*2, 128, 128) -> (N, out_channels, 256, 256) (decoded_4+initial)

	def forward(self, x):

		initial = self.initial_block(x)
		encoded_1 = self.encoder_block_1(initial)
		encoded_2 = self.encoder_block_2(encoded_1)
		encoded_3 = self.encoder_block_3(encoded_2)
		encoded_4 = self.encoder_block_4(encoded_3)

		bottleneck = self.bottleneck_block(encoded_4)

		decoded_1 = self.decoder_block_1(bottleneck, encoded_4)
		decoded_2 = self.decoder_block_2(decoded_1, encoded_3)
		decoded_3 = self.decoder_block_3(decoded_2, encoded_2)
		decoded_4 = self.decoder_block_4(decoded_3, encoded_1)

		output = self.final_block(torch.cat([decoded_4, initial], dim=1))

		activations = [initial.detach(), encoded_1.detach(), encoded_2.detach(), encoded_3.detach(), encoded_4.detach(), bottleneck.detach(), decoded_1.detach(), decoded_2.detach(), decoded_3.detach(), decoded_4.detach()]
		
		return output, activations

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
	x = torch.randn((1, 4, 256, 256))
	print('x:', x.shape)
	model = Generator(in_channels=4, out_channels=1)
	pred, _ = model(x)
	print('output:', pred.shape)

if __name__ == "__main__":
	test()