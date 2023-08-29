import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math

def train_gan(device, training_dataloader, discriminator, generator, gen_lr, disc_lr, l1_lambda, visualization_frequency):

	bce = nn.BCELoss()
	l1 = nn.L1Loss()
	gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)
	disc_optimizer = optim.SGD(discriminator.parameters(), lr=disc_lr)

	generator.train()
	discriminator.train()

	for step, (sample, target) in enumerate(training_dataloader):

		sample = sample.to(device) # sketch
		target = target.to(device) # real dem
		fake, activations = generator(sample)

		# Train the discriminator
		disc_pred_real = discriminator(sample, target)
		disc_loss_real = bce(disc_pred_real, torch.ones_like(disc_pred_real))
		disc_optimizer.zero_grad()
		disc_loss_real.backward()
		disc_optimizer.step()

		disc_pred_fake = discriminator(sample, fake.detach())
		disc_loss_fake = bce(disc_pred_fake, torch.zeros_like(disc_pred_fake))
		disc_optimizer.zero_grad()
		disc_loss_fake.backward()
		disc_optimizer.step()

		# Train the generator
		disc_pred_generated = discriminator(sample, fake)
		gen_loss_fake = bce(disc_pred_generated, torch.ones_like(disc_pred_generated))
		gen_loss_l1 = l1(fake, target) * l1_lambda
		gen_total_loss = gen_loss_fake + gen_loss_l1
		
		gen_optimizer.zero_grad()
		gen_total_loss.backward()
		gen_optimizer.step()

		wandb.log({
			'loss/disc_real': disc_loss_real,
			'loss/disc_fake': disc_loss_fake,
			'loss/gen': gen_total_loss
		})

		if (step % visualization_frequency == 0):
			print(f"Generating visulizations for training sample {step}")
			visualize_example(fake.detach().cpu())
			# visualize_activations(activations)
			# visualize_kernels(generator)
			print("Back to training")

def visualize_example(examples):
	# print("Visualizing example...")
	rows = int(math.sqrt(examples.shape[0]))
	grid = torchvision.utils.make_grid(examples, nrow=rows, normalize=True, padding=2)
	image = wandb.Image(grid, caption="generated")
	wandb.log({f"inference": image})


def visualize_activations(activations):
	# print("Visualizing activations...")
	for index, activation in enumerate(activations):
		n, c, h, w = activation.shape
		tensor = activation.cpu().view(n*c, -1, h, w)
		rows = int(math.sqrt(tensor.shape[0]))
		grid = torchvision.utils.make_grid(tensor, nrow=rows, normalize=True, padding=1)
		image = wandb.Image(grid)
		wandb.log({f"activations/{index}": image})

def visualize_kernels(model):
	save_layer_visualization('initial_block', model.initial_block[0], 16)
	save_layer_visualization('encoder_1', model.encoder_block_1.conv[0], 32)
	save_layer_visualization('encoder_2', model.encoder_block_2.conv[0], 64)
	save_layer_visualization('encoder_3', model.encoder_block_3.conv[0], 128)
	save_layer_visualization('encoder_4', model.encoder_block_4.conv[0], 256)
	save_layer_visualization('encoder_5', model.encoder_block_5.conv[0], 256)
	save_layer_visualization('bottleneck', model.bottleneck_block[0], 256)
	save_layer_visualization('decoder_1', model.decoder_block_1.conv[0], 256)
	save_layer_visualization('decoder_2', model.decoder_block_2.conv[0], 128)
	save_layer_visualization('decoder_3', model.decoder_block_3.conv[0], 64)
	save_layer_visualization('decoder_4', model.decoder_block_4.conv[0], 32)
	save_layer_visualization('decoder_5', model.decoder_block_5.conv[0], 16)
	save_layer_visualization('final_block', model.final_block[0], 16)

def save_layer_visualization(name, layer, num_rows=8, padding=1):
	print(f"Visualizing kernels for layer {name}")
	kernels = layer.weight.detach().clone()
	n,c,w,h = kernels.shape
	kernels = kernels.view(n*c, -1, w, h)
	grid = torchvision.utils.make_grid(kernels, nrow=num_rows, normalize=True, padding=padding).cpu()
	image = wandb.Image(grid, caption=name)
	wandb.log({f"kernels/{name}": image})
