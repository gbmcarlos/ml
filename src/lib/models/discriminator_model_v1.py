import torch
import torch.nn as nn


class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.2)
		)

	def forward(self, x):
		return self.conv(x)

# Starting with x like (N, 3, 256, 256) and y like (N, 1, 256, 256), output a z like (N, 1, 7, 7)
class Discriminator(nn.Module):
	def __init__(self, in_channels_x, in_channels_y):
		super().__init__()

		self.initial_block = nn.Sequential(
			nn.Conv2d(in_channels_x + in_channels_y, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
			nn.ReLU()
		)# (N, in_channels_x + in_channels_y, 256, 256) -> (N, 64, 128, 128)

		self.cnn_block_1 = CNNBlock(in_channels=64, out_channels=64) # (N, 64, 128, 128) -> (N, 64, 64, 64)
		self.cnn_block_2 = CNNBlock(in_channels=64, out_channels=128) # (N, 64, 64, 64) -> (N, 128, 32, 32)
		self.cnn_block_3 = CNNBlock(in_channels=128, out_channels=256) # (N, 128, 32, 32) -> (N, 256, 16, 16)
		self.cnn_block_4 = CNNBlock(in_channels=256, out_channels=512) # (N, 256, 16, 16) -> (N, 512, 8, 8)

		self.final_block = nn.Sequential(
			nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'), # (N, 512, 8, 8) -> (N, 1, 7, 7)
			nn.Sigmoid()
		)

	def forward(self, x, y):
		out = torch.cat([x, y], dim=1) # Concatenate the input and the target, along the channels
		out = self.initial_block(out)
		out = self.cnn_block_1(out)
		out = self.cnn_block_2(out)
		out = self.cnn_block_3(out)
		out = self.cnn_block_4(out)
		out = self.final_block(out)
		return out
	
def test():
	x = torch.randn((1, 3, 256, 256))
	y = torch.randn((1, 1, 256, 256))
	print('x:', x.shape)
	print('y:', y.shape)
	model = Discriminator(3, 1)
	pred = model(x, y)
	print('output:', pred.shape)

if __name__ == "__main__":
	test()