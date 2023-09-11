import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from schema import Schema
import os
import wandb
import sys, signal

from ..trainers.WGanTrainer import WGanTrainer
from ..trainers.GGanTrainer import GGanTrainer
from ..models import generator_model_v1 as generator_model, discriminator_model_v1 as discriminator_model
from ..services import dataset_service


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
				'critic_lr': float,
				'betas': list,
				'l1_lambda': int,
				'gp_lambda': int,
				'critic_iter': int
			}
		}
	}, ignore_extra_keys=True)
	config_schema.validate(settings)
	name = settings['name']
	settings = settings['train']
	hyper = settings['hyper']

	device = torch.device(settings['device_name'])
	print('Loading training data...')
	
	training_dataloader = get_dataset(
		settings['training_data_folder'], 
		settings['tile_filter_prefix'],
		hyper['batch_size']
	)

	generator = generator_model.Generator(in_channels=4, out_channels=1).to(device)
	critic = discriminator_model.Discriminator(in_channels_x=4, in_channels_y=1).to(device)
	generator_model.initialize_weights(generator)
	generator_model.initialize_weights(critic)

	trainer = WGanTrainer(
		generator, critic,
		"src/data/checkpoints", hyper, device, training_dataloader, settings['visualization_frequency'], 
		# True
	)
	# trainer = GGanTrainer(
		# generator_1, generator_2, discriminator,
		# "src/data/checkpoints", hyper, device, training_dataloader, settings['visualization_frequency'],
		# True
	# )

	trainer.run(name)

	return

def get_dataset(data_folder, tile_filter_prefix, batch_size):

	training_dataset = dataset_service.GanDataset(data_folder, tile_filter_prefix)
	# train_set, validation_set = torch.utils.data.random_split(training_dataset, [0.7, 0.3])
	training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
	# validation_dataloader = DataLoader(validation_set, batch_size=1, shuffle=False)

	sketch, dem = next(iter(training_dataloader))
	print('sketch ->', 'type:', type(sketch), 'dtype:', sketch.dtype, 'shape:', sketch.shape, 'range:', [sketch.min(), sketch.max()])
	print('dem	->', 'type:', type(dem), 'dtype:', dem.dtype, 'shape:', dem.shape, 'range:', [dem.min(), dem.max()])

	return training_dataloader