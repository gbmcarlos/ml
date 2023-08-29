import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.2)
		)

	def forward(self, x):
		return self.conv(x)

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, dropout=0.0):
		super().__init__()

		self.conv = nn.Sequential(
			nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.Dropout(dropout),
			nn.ReLU(),
		)

	def forward(self, x, y=None):
		if y is not None:
			x = torch.cat([x, y], dim=1)
		return self.conv(x)


# Starting with x like (N, 3, 256, 256), output a z like (N, 1, 256, 256)
class Generator(nn.Module):
	def __init__(self, in_channels=3, out_channels=1):
		super().__init__()

		self.initial_block = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
			nn.BatchNorm2d(64),
		) # initial: (N, in_channels, 256, 256) -> (N, 64, 128, 128)

		self.encoder_block_1 = EncoderBlock(64, 128) # encoded_1: (N, 64, 128, 128)-> (N, 128, 64, 64)
		self.encoder_block_2 = EncoderBlock(128, 256) # encoded_2: (N, 128, 64, 64) -> (N, 256, 32, 32)
		self.encoder_block_3 = EncoderBlock(256, 512) # encoded_3: (N, 256, 32, 32) -> (N, 512, 16, 16)
		self.encoder_block_4 = EncoderBlock(512, 512) # encoded_4: (N, 512, 16, 16) -> (N, 512, 8, 8)
		self.encoder_block_5 = EncoderBlock(512, 512) # encoded_5: (N, 512, 8, 8) -> (N, 512, 4, 4)

		self.bottleneck_block = nn.Sequential(
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding='same', padding_mode='reflect', bias=False),
			nn.BatchNorm2d(512),
		) # bottleneck: (N, 512, 4, 4) -> (N, 512, 4, 4) (=encoded_7 in pix2pix)

		self.decoder_block_1 = DecoderBlock(512, 512, 0.5) # decoded_1: (N, 512, 4, 4) -> (N, 512, 8, 8) (bottleneck+none) (=decoded_7 in pix2pix)
		self.decoder_block_2 = DecoderBlock(512*2, 512, 0.5) # decoded_2: (N, 512*2, 8, 8) -> (N, 512, 16, 16) (decoded_1+encoded_4)
		self.decoder_block_3 = DecoderBlock(512*2, 256) # decoded_3: (N, 512*2, 16, 16) -> (N, 256, 32, 32) (decoded_2+encoded_3)
		self.decoder_block_4 = DecoderBlock(256*2, 128) # decoded_4: (N, 256*2, 32, 32) -> (N, 128, 64, 64) (decoded_3+encoded_2)
		self.decoder_block_5 = DecoderBlock(128*2, 64) # decoded_5: (N, 128*2, 64, 64) -> (N, 64, 128, 128) (decoded_4+encoded_1)

		self.output_block = nn.Sequential(
			nn.ConvTranspose2d(in_channels=64*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh()
		) # output_block: (N, 128*2, 128, 128) -> (N, out_channels, 256, 256) (decoded_5+initial)

	def forward(self, x):

		initial = self.initial_block(x)

		encoded_1 = self.encoder_block_1(initial)
		encoded_2 = self.encoder_block_2(encoded_1)
		encoded_3 = self.encoder_block_3(encoded_2)
		encoded_4 = self.encoder_block_4(encoded_3)
		encoded_5 = self.encoder_block_5(encoded_4)

		bottleneck = self.bottleneck_block(encoded_5)

		decoded_1 = self.decoder_block_1(bottleneck)
		decoded_2 = self.decoder_block_2(decoded_1, encoded_4)
		decoded_3 = self.decoder_block_3(decoded_2, encoded_3)
		decoded_4 = self.decoder_block_4(decoded_3, encoded_2)
		decoded_5 = self.decoder_block_5(decoded_4, encoded_1)

		output = self.output_block(torch.cat([decoded_5, initial], dim=1))

		activations = [initial.detach(), encoded_1.detach(), encoded_2.detach(), encoded_3.detach(), encoded_4.detach(), encoded_5.detach(), bottleneck.detach(), decoded_1.detach(), decoded_2.detach(), decoded_3.detach(), decoded_4.detach(), decoded_5.detach()]
		
		return output, activations

def test():
	x = torch.randn((1, 3, 256, 256))
	print('x:', x.shape)
	model = Generator()
	pred, _ = model(x)
	print('output:', pred.shape)

if __name__ == "__main__":
	test()