import torch, torchvision, wandb
import signal, time, calendar, os


class WGanTrainer():
	def __init__(self, device, generator, critic, dataloader, hyperparameters, visualization_frequency, dry_run=False):
		self.device = device
		self.generator = generator
		self.critic = critic
		self.dataloder = dataloader
		self.hyperparameters = hyperparameters
		self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hyperparameters['gen_lr'], betas=self.hyperparameters['betas'])
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyperparameters['critic_lr'], betas=(0.0, 0.9))
		self.l1 = torch.nn.L1Loss()
		self.visualization_frequency = visualization_frequency
		self.dry_run = dry_run
		self.stopped = False
		self.paused = False
		self.init_signals()
	
	def run(self, run_name):

		self.load()

		if not self.dry_run:
			wandb.init(
				project='terramorph',
				config=self.hyperparameters,
				name=run_name
			)
			# wandb.watch(generator, log='all', log_freq=1, idx=0)
			# wandb.watch(discriminator, log='all', log_freq=1, idx=1)
		
		for index, epoch in enumerate(range(self.hyperparameters['epochs'])):

			print(f"Starting epoch {epoch+1}")
			
			for step, (sample, target) in enumerate(self.dataloder):

				self.step(epoch, step, sample, target)

				if not self.should_continue():
					print('Stopped epoch')
					break
			
			if not self.should_continue():
				print('Stopped execution')
				break

	def step(self, epoch, step, sample, target):

		sample = sample.to(self.device) # sketch
		target = target.to(self.device) # dem
		
		for _ in range(self.hyperparameters['critic_iter']):
			
			fake, _ = self.generator(sample)
			critic_pred_real = self.critic(sample, target).reshape(-1)
			critic_pred_fake = self.critic(sample, fake).reshape(-1)
			critic_loss_gp = self.gradient_penalty(sample, target, fake) * self.hyperparameters['gp_lambda']
			critic_loss_w_distance = torch.mean(critic_pred_real) - torch.mean(critic_pred_fake)
			critic_loss_total = (
				- critic_loss_w_distance # Optimizers minimize, but we want to maximize, so negative
				+ critic_loss_gp
			)
			self.critic.zero_grad()
			critic_loss_total.backward(retain_graph=True)
			self.critic_optimizer.step()

		critic_pred_generated = self.critic(sample, fake).reshape(-1)
		gen_loss_w = -torch.mean(critic_pred_generated)
		gen_loss_l1 = self.l1(fake, target) * self.hyperparameters['l1_lambda']
		gen_loss_total = gen_loss_w + gen_loss_l1
		self.generator.zero_grad()
		gen_loss_total.backward()
		self.gen_optimizer.step()

		if not self.dry_run:
			wandb.log({
				'loss/critic_w': critic_loss_w_distance,
				'loss/critic_gp': critic_loss_gp,
				'loss/gen_w': gen_loss_w,
				'loss/gen_l1': gen_loss_l1
			})

		if (step % self.visualization_frequency == 0):
			print(f"Generating visulizations for step {step}")
			self.visualize_batch(fake.detach().cpu(), 'inference')
	
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

	def visualize_batch(self, batch, name):
		grid = self.get_image_grid(batch, 512)
		image = wandb.Image(grid)
		if not self.dry_run:
			wandb.log({name: image})

	def get_image_grid(self, batch, max_size=512):
		n, c, h, w = batch.shape
		columns = int(max_size/w) # How many of these "images" we could fit side-to-side in a max_size width
		size = int(min(columns*columns, n*c)) # Don't try to select more than available, even if they could fit
		layers = batch.cpu().view(n*c, -1, h, w) # Unpack the channels, so a long list of images
		indices = torch.randperm(n*c)[:size] # Get a shuffled list of indices and select as many as required
		images = layers[indices]
		grid = torchvision.utils.make_grid(images, nrow=columns, normalize=True, padding=2)
		return grid

	def save(self):
		print('Saving models...')
		current_gmt = time.gmtime()
		timestamp = calendar.timegm(current_gmt)

		checkpoint = {
			"generator": self.generator.state_dict(),
			"generator_optimizer": self.gen_optimizer.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict()
		}
		torch.save(checkpoint, os.path.join("src/data/checkpoints", "checkpoint_" + str(timestamp) + ".tar"))
	
	def load(self):
		
		checkpoints = os.listdir("src/data/checkpoints")
		if not checkpoints:
			print('No checkpoints found')
			return

		latest = sorted(checkpoints, reverse=True)[0]
		confirmation = input(f"Latest checkpoint {latest} found. Load? [y/n] ").lower()
		if confirmation != 'y':
			print('Skipped loading models from checkpoint')
			return
		
		print('Loading checkpoint')
		checkpoint = torch.load(os.path.join("src/data/checkpoints", latest), map_location=self.device)
		self.generator.load_state_dict(checkpoint["generator"])
		self.gen_optimizer.load_state_dict(checkpoint["generator_optimizer"])
		self.critic.load_state_dict(checkpoint["critic"])
		self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

		for param_group in self.gen_optimizer.param_groups:
			param_group['lr'] = self.hyperparameters['gen_lr']
		
		for param_group in self.critic_optimizer.param_groups:
			param_group['lr'] = self.hyperparameters['critic_lr']

	def should_continue(self):

		if self.stopped:
			return False

		if self.paused:
			action = input("Action [stop, resume, save]: ").lower()

			if (action == 'stop'):
				self.stopped = True
				return False
			elif action == 'resume':
				self.paused = False
				return True
			elif action == 'save':
				self.save()
				return self.should_continue()
		
		return True

	def init_signals(self):
		signal.signal(signal.SIGINT, self.interrupt)

	def interrupt(self, signum, frame):
		self.paused = True
