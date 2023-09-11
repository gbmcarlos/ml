import torch, torchvision, wandb
import signal, time, calendar, os
from .BaseGanTrainer import BaseGanTrainer


class WGanTrainer(BaseGanTrainer):
	def __init__(self, generator, critic, checkpoint_folder, hyperparameters, device, dataloader, visualization_frequency, dry_run=False):
		super().__init__(checkpoint_folder, hyperparameters, device, dataloader, visualization_frequency, dry_run)
		self.generator = generator
		self.critic = critic
		self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hyperparameters['gen_lr'], betas=self.hyperparameters['betas'])
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyperparameters['critic_lr'], betas=(0.0, 0.9))
		self.l1 = torch.nn.L1Loss()

	def step(self, epoch, step, sample, target):

		sample = sample.to(self.device) # sketch
		target = target.to(self.device) # dem
		
		for _ in range(self.hyperparameters['critic_iter']):

			fake, _ = self.generator(sample)

			critic_pred_real = self.critic(sample, target).reshape(-1)
			critic_pred_fake = self.critic(sample, fake).reshape(-1)
			critic_loss_real = -critic_pred_real.mean() # Maximize the mean for real
			critic_loss_fake = critic_pred_fake.mean() # Minimize the mean for fake
			critic_loss_gp = self.gradient_penalty(sample, target, fake) * self.hyperparameters['gp_lambda']
			critic_loss_total = (
				critic_loss_real
				+critic_loss_fake
				+critic_loss_gp
			)
			self.critic.zero_grad()
			critic_loss_total.backward(retain_graph=True)
			self.critic_optimizer.step()
		
		critic_pred_generated = self.critic(sample, fake).reshape(-1)
		gen_loss_fake = -torch.mean(critic_pred_generated) # Maximize the mean for fake
		# gen_loss_l1 = self.l1(fake, target) * self.hyperparameters['l1_lambda']
		# gen_loss_total = gen_loss_w
		self.generator.zero_grad()
		gen_loss_fake.backward()
		self.gen_optimizer.step()

		self.wandb_log({
			'loss/critic_real': critic_loss_real,
			'loss/critic_fake': critic_loss_fake,
			'loss/critic_gp': critic_loss_gp,
			'loss/gen_fake': gen_loss_fake,
			# 'loss/gen_l1': gen_loss_l1
		})

		if (step % self.visualization_frequency == 0):
			print(f"Generating visulizations for step {step}")
			self.visualize_batch(fake.detach().cpu(), 'inference')
	
	def get_checkpoint(self):
		return {
			"generator": self.generator.state_dict(),
			"generator_optimizer": self.gen_optimizer.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict()
		}

	def apply_checkpoint(self, checkpoint):
		self.generator.load_state_dict(checkpoint["generator"])
		self.gen_optimizer.load_state_dict(checkpoint["generator_optimizer"])
		self.critic.load_state_dict(checkpoint["critic"])
		self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

		for param_group in self.gen_optimizer.param_groups:
			param_group['lr'] = self.hyperparameters['gen_lr']
		
		for param_group in self.critic_optimizer.param_groups:
			param_group['lr'] = self.hyperparameters['critic_lr']

	def gradient_penalty(self, sample, target, fake):
		N, C, H, W = target.shape
		alpha = torch.rand((N, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
		interpolated_images = target * alpha + fake * (1 - alpha)

		# Calculate critic scores
		mixed_scores = self.critic(sample, interpolated_images)

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
