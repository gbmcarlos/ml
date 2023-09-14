import torch, torchvision, wandb
import signal, time, calendar, os
from .BaseGanTrainer import BaseGanTrainer


class LSGanTrainer(BaseGanTrainer):
	def __init__(self, generator, discriminator, checkpoint_folder, hyperparameters, device, dataloader, visualization_frequency, dry_run=False):
		super().__init__(checkpoint_folder, hyperparameters, device, dataloader, visualization_frequency, dry_run)
		self.generator = generator
		self.discriminator = discriminator
		self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hyperparameters['gen_lr'], betas=self.hyperparameters['betas'])
		self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hyperparameters['disc_lr'], betas=(0.0, 0.9))
		self.bce = torch.nn.BCELoss()
		self.l1 = torch.nn.L1Loss()

	def step(self, epoch, step, sample, target):

		sample = sample.to(self.device) # sketch
		target = target.to(self.device) # real dem
		fake, activations = self.generator(sample)

		# Train the discriminator
		self.disc_optimizer.zero_grad()

		disc_pred_real = self.discriminator(sample, target)
		labels_real = torch.ones_like(disc_pred_real)
		disc_loss_real = 0.5 * torch.mean((disc_pred_real-labels_real)**2)
		disc_loss_real.backward()

		disc_pred_fake = self.discriminator(sample, fake.detach())
		labels_fake = torch.zeros_like(disc_pred_real)
		disc_loss_fake = 0.5 * torch.mean((disc_pred_fake-labels_fake)**2)
		disc_loss_fake.backward()

		self.disc_optimizer.step()

		# Train the generator
		self.gen_optimizer.zero_grad()

		disc_pred_generated = self.discriminator(sample, fake)
		labels_generated = torch.ones_like(disc_pred_real)
		gen_loss_fake = 0.5 * torch.mean((disc_pred_generated-labels_generated)**2)
		gen_loss_fake.backward(retain_graph=True)

		gen_loss_l1 = self.l1(fake, target) * self.hyperparameters['l1_lambda']
		gen_loss_l1.backward()
		self.gen_optimizer.step()

		self.wandb_log({
			'loss/disc_real': disc_loss_real,
			'loss/disc_fake': disc_loss_fake,
			'loss/gen_l1': gen_loss_l1,
			'loss/gen_fake': gen_loss_fake
		})

		if (step % self.visualization_frequency == 0):
			print(f"Generating visulizations for step {step}")
			self.visualize_batch(fake.detach().cpu(), 'inference')
	
	def get_checkpoint(self):
		return {
			"generator": self.generator.state_dict(),
			"generator_optimizer": self.gen_optimizer.state_dict(),
			"discriminator": self.discriminator.state_dict(),
			"discriminator_optimizer": self.disc_optimizer.state_dict()
		}

	def apply_checkpoint(self, checkpoint):
		self.generator.load_state_dict(checkpoint["generator"])
		self.gen_optimizer.load_state_dict(checkpoint["generator_optimizer"])
		self.discriminator.load_state_dict(checkpoint["discriminator"])
		self.disc_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])

		for param_group in self.gen_optimizer.param_groups:
			param_group['lr'] = self.hyperparameters['gen_lr']
		
		for param_group in self.disc_optimizer.param_groups:
			param_group['lr'] = self.hyperparameters['disc_lr']
