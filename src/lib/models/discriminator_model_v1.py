import torch
import torch.nn as nn


class CNNBlock(nn.Module):
	def __init__(self, norm, in_channels, out_channels):
		super().__init__()

		layers = [
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
		]

		if norm == "batch":
			layers.append(nn.BatchNorm2d(out_channels))
		elif norm == "inst":
			layers.append(nn.InstanceNorm2d(out_channels))
		
		layers.append(nn.LeakyReLU(0.2))
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		return self.conv(x)


class Discriminator(nn.Module):
	def __init__(self, in_channels_x, in_channels_y, norm="batch", output="sigmoid"):
		super().__init__()

		self.initial_block = CNNBlock(norm, in_channels=in_channels_x + in_channels_y, out_channels=64) # initial: (N, in_channels_x + in_channels_y, 256, 256) -> (N, 64, 128, 128)
		self.cnn_block_1 = CNNBlock(norm, in_channels=64, out_channels=128) # (N, 64, 128, 128) -> (N, 128, 64, 64)
		self.cnn_block_2 = CNNBlock(norm, in_channels=128, out_channels=256) # (N, 128, 64, 64) -> (N, 256, 32, 32)
		self.cnn_block_3 = CNNBlock(norm, in_channels=256, out_channels=512) # (N, 256, 32, 32) -> (N, 512, 16, 16)
		self.cnn_block_4 = CNNBlock(norm, in_channels=512, out_channels=1024) # (N, 512, 16, 16) -> (N, 1024, 8, 8)
		self.cnn_block_5 = CNNBlock(norm, in_channels=1024, out_channels=2048) # (N, 1024, 8, 8) -> (N, 2048, 4, 4)

		final_block_layers = [
			nn.Conv2d(2048, 1, kernel_size=4, stride=2, padding=0), # (N, 2048, 4, 4) -> (N, 1, 1, 1)
		]

		if output == "sigmoid":
			final_block_layers.append(nn.Sigmoid())
		elif output == "flatten":
			final_block_layers.append(nn.Flatten(1)) # (N, 1, 1, 1) -> (N, 1)
		
		self.final_block = nn.Sequential(*final_block_layers)

	def forward(self, x, y):
		out = torch.cat([x, y], dim=1) # Concatenate the input and the target, along the channels
		
		initial = self.initial_block(out)
		conv_1 = self.cnn_block_1(initial)
		conv_2 = self.cnn_block_2(conv_1)
		conv_3 = self.cnn_block_3(conv_2)
		conv_4 = self.cnn_block_4(conv_3)
		conv_5 = self.cnn_block_5(conv_4)
		out = self.final_block(conv_5)

		# activations = [initial.detach(), conv_1.detach(), conv_2.detach(), conv_3.detach()]

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