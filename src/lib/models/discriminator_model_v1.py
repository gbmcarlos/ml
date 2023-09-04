import torch
import torch.nn as nn


class CNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
			nn.InstanceNorm2d(out_channels, affine=True),
			nn.LeakyReLU(0.2)
		)

	def forward(self, x):
		return self.conv(x)


# Starting with x like (N, 3, 256, 256) and y like (N, 1, 256, 256), output a z like (N, 1, 7, 7)
class Discriminator(nn.Module):
	def __init__(self, in_channels_x, in_channels_y):
		super().__init__()

		self.initial_block = nn.Sequential(
			nn.Conv2d(in_channels_x + in_channels_y, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
		) # initial: (N, in_channels_x + in_channels_y, 256, 256) -> (N, 64, 128, 128)

		self.cnn_block_1 = CNNBlock(in_channels=64, out_channels=128) # (N, 64, 128, 128) -> (N, 128, 64, 64)
		self.cnn_block_2 = CNNBlock(in_channels=128, out_channels=256) # (N, 128, 64, 64) -> (N, 256, 32, 32)
		self.cnn_block_3 = CNNBlock(in_channels=256, out_channels=512) # (N, 256, 32, 32) -> (N, 512, 16, 16)
		self.cnn_block_4 = CNNBlock(in_channels=512, out_channels=512) # (N, 512, 16, 16) -> (N, 512, 8, 8)
		self.cnn_block_5 = CNNBlock(in_channels=512, out_channels=512) # (N, 512, 8, 8) -> (N, 512, 4, 4)

		self.final_block = nn.Sequential(
			nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0),
		) # (N, 512, 8, 8) -> (N, 1, 1, 1)

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
	
def gradient_penalty(critic, sample, target, fake, device="cpu"):
	N, C, H, W = target.shape
	alpha = torch.rand((N, 1, 1, 1)).repeat(1, C, H, W).to(device)
	interpolated_images = target * alpha + fake * (1 - alpha)

    # Calculate critic scores
	mixed_scores = critic(sample, interpolated_images)

    # Take the gradient of the scores with respect to the images
	gradient = torch.autograd.grad(
		inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
	gradient = gradient.view(gradient.shape[0], -1)
	gradient_norm = gradient.norm(2, dim=1)
	gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
	return gradient_penalty

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