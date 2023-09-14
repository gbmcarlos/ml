import os, signal, time, calendar
from abc import ABC, abstractmethod
import wandb, torch, torchvision


class BaseGanTrainer:
	def __init__(self, checkpoint_folder, hyperparameters, device, dataloader, visualization_frequency, dry_run=False) -> None:
		self.checkpoint_folder = checkpoint_folder
		self.hyperparameters = hyperparameters
		self.device = device
		self.dataloder = dataloader
		self.visualization_frequency = visualization_frequency
		self.dry_run = dry_run
		self.stopped = False
		self.paused = False
		self.init_signals()

	@abstractmethod
	def step(self, epoch, step, sample, target):
		pass

	@abstractmethod
	def get_checkpoint(self):
		pass

	@abstractmethod
	def apply_checkpoint(self, checkpoint):
		pass

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
		current_gmt = time.gmtime()
		timestamp = calendar.timegm(current_gmt)
		name = "checkpoint_" + str(timestamp) + ".tar"
		print(f'Saving models to {name}')

		# description = input("Checkpoint description: ")

		checkpoint = self.get_checkpoint()

		torch.save(checkpoint, os.path.join(self.checkpoint_folder, name))


	def load(self):
		
		checkpoints = os.listdir(self.checkpoint_folder)
		if not checkpoints:
			print('No checkpoints found')
			return

		latest = sorted(checkpoints, reverse=True)[0]
		confirmation = input(f"Latest checkpoint {latest} found. Load? [y/n] ").lower()
		if confirmation != 'y':
			print('Skipped loading models from checkpoint')
			return
		
		print('Loading checkpoint')
		checkpoint = torch.load(os.path.join(self.checkpoint_folder, latest), map_location=self.device)

		self.apply_checkpoint(checkpoint)

		return checkpoint

	def wandb_log(self, log):
		if not self.dry_run:
			wandb.log(log)

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