import argparse, yaml
from schema import Schema, Optional
import torch
from torch.utils.data import DataLoader

from lib.services import dataset_service
from lib.models import generator_model_v1 as generator_model, discriminator_model_v1 as discriminator_model
from lib.trainers.WGanTrainer import WGanTrainer
from lib.trainers.LSGanTrainer import LSGanTrainer


def run():

	parser = argparse.ArgumentParser(description="Train a Neural Network on DEM-Sketch pairs")
	parser.add_argument('--settings-path', type=str, required=True)
	parser.add_argument('--name', type=str, required=False)
	args = parser.parse_args()
	settings = read_settings(args.settings_path)
	settings['name'] = args.name

	train_gan(settings)

	return

def read_settings(settings_path):
	try:
		with open(settings_path) as file:
			settings = yaml.load(file, Loader=yaml.FullLoader)
			return settings
	except Exception as e:
		print(f"An error occurred while reading the settings file: {e}")

def train_gan(settings):

	config_schema = Schema({
		'train': {
			'training_data_folder': str,
			'device_name': str,
			'visualization_frequency': int,
			'hyper': {
				'batch_size': int,
				'epochs': int,
				'gen_lr': float,
				'disc_lr': float,
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
		hyper['batch_size']
	)

	generator = generator_model.Generator(in_channels=4, out_channels=1).to(device)
	discriminator = discriminator_model.Discriminator(in_channels_x=4, in_channels_y=1, norm="batch", output="sigmoid").to(device)
	generator_model.initialize_weights(generator)
	generator_model.initialize_weights(discriminator)

	trainer = LSGanTrainer(
		generator, discriminator,
		"src/data/checkpoints", hyper, device, training_dataloader, settings['visualization_frequency'], 
		# True
	)

	trainer.run(name)

	return

def get_dataset(data_folder, batch_size):

	training_dataset = dataset_service.GanDataset(data_folder)
	training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

	sketch, dem = next(iter(training_dataloader))
	print('sketch ->', 'type:', type(sketch), 'dtype:', sketch.dtype, 'shape:', sketch.shape, 'range:', [sketch.min(), sketch.max()])
	print('dem	->', 'type:', type(dem), 'dtype:', dem.dtype, 'shape:', dem.shape, 'range:', [dem.min(), dem.max()])

	return training_dataloader

if __name__ == '__main__':
	run()
