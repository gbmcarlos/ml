import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from schema import Schema
import os
import wandb

from ..models import generator_model_v2 as generator_model, discriminator_model_v2 as discriminator_model
from ..services import dataset_service, train_service, checkpoint_service


def train_gan(settings):

	config_schema = Schema({
		'train': {
			'training_data_folder': str,
			'tile_filter_prefix': str,
			'device_name': str,
			'visualization_frequency': int,
			'hyper': {
				'batch_size': int,
				'epochs': int,
				'gen_lr': float,
				'disc_lr': float,
				'betas': list,
				'l1_lambda': float
			}
		}
	}, ignore_extra_keys=True)
	config_schema.validate(settings)
	settings = settings['train']
	hyper = settings['hyper']

	wandb.init(
		project='terramorph',
		config=hyper
	)

	device = torch.device(settings['device_name'])
	print('Loading training data...')
	
	training_dataloader = get_dataset(
		settings['training_data_folder'], 
		settings['tile_filter_prefix'],
		hyper['batch_size']
	)

	discriminator = discriminator_model.Discriminator(in_channels_x=3, in_channels_y=1).to(device)
	generator = generator_model.Generator(in_channels=3, out_channels=1).to(device)

	# gen_checkpoint_path = os.path.join(settings['checkpoint_folder'], 'gen.tar')
	# disc_checkpoint_path = os.path.join(settings['checkpoint_folder'], 'disc.tar')

	print('Loading checkpoints')
	# checkpoint_service.load_checkpoint(generator, gen_optimizer, device, settings['learning_rate'], gen_checkpoint_path)
	# checkpoint_service.load_checkpoint(discriminator, disc_optimizer, device, settings['learning_rate'], disc_checkpoint_path)

	print(f"Training for {hyper['epochs']} epochs")
	for index, epoch in enumerate(range(hyper['epochs'])):
		print(f"Starting epoch {epoch+1}")
		train_service.train_gan(
			device, training_dataloader, discriminator, generator,
			hyper['gen_lr'], hyper['disc_lr'], hyper['betas'], hyper['l1_lambda'],
			visualization_frequency=settings['visualization_frequency']
		)

		print('Saving checkpoints')
		# checkpoint_service.save_checkpoint(generator, gen_optimizer, gen_checkpoint_path)
		# checkpoint_service.save_checkpoint(discriminator, disc_optimizer, disc_checkpoint_path)

def get_dataset(data_folder, tile_filter_prefix, batch_size):

	training_dataset = dataset_service.GanDataset(data_folder, tile_filter_prefix)
	# train_set, validation_set = torch.utils.data.random_split(training_dataset, [0.7, 0.3])
	training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
	# validation_dataloader = DataLoader(validation_set, batch_size=1, shuffle=False)

	dem, sketch = next(iter(training_dataloader))
	print('sketch ->', 'type:', type(sketch), 'dtype:', sketch.dtype, 'shape:', sketch.shape, 'range:', [sketch.min(), sketch.max()])
	print('dem	->', 'type:', type(dem), 'dtype:', dem.dtype, 'shape:', dem.shape, 'range:', [dem.min(), dem.max()])

	return training_dataloader