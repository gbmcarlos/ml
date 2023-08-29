import torch
import torch.nn as nn


class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=2, padding=1, bias=False, padding_mode='reflect'):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=padding, padding_mode=padding_mode, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.2)
		)

	def forward(self, x):
		return self.conv(x)

# Starting with x like (N, 3, 256, 256) and y like (N, 1, 256, 256), output a z like (N, 1, 8, 8)
class Discriminator(nn.Module):
	def __init__(self, in_channels_x, in_channels_y):
		super().__init__()

		self.initial_block = nn.Sequential(
			nn.Conv2d(in_channels_x + in_channels_y, out_channels=64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2)
		) # initial: (N, in_channels_x + in_channels_y, 256, 256) -> (N, 64, 128, 128)

		self.cnn_block_1 = CNNBlock(in_channels=64, out_channels=128) # conv_1: (N, 64, 128, 128) -> (N, 128, 64, 64)
		self.cnn_block_2 = CNNBlock(in_channels=128, out_channels=256) # conv_2: (N, 128, 64, 64) -> (N, 256, 32, 32)
		self.cnn_block_3 = CNNBlock(in_channels=256, out_channels=512) # conv_3: (N, 256, 32, 32) -> (N, 512, 16, 16)

		self.final_block = nn.Sequential(
			nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
			nn.Sigmoid()
		) # output: (N, 512, 16, 16) -> (N, 1, 8, 8)

	def forward(self, x, y):
		input = torch.cat([x, y], dim=1) # Concatenate the input and the target, along the channels
		initial = self.initial_block(input)
		conv_1 = self.cnn_block_1(initial)
		conv_2 = self.cnn_block_2(conv_1)
		conv_3 = self.cnn_block_3(conv_2)
		output = self.final_block(conv_3)
		return output
	
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