import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from schema import Schema
import os

from ..models import generator_model, discriminator_model
from ..services import dataset_service, train_service, checkpoint_service


def train_gan(settings):

    config_schema = Schema({
        'train': {
            'training_data_folder': str,
            'tile_filter_prefix': str,
            'examples_folder': str,
            'flow_threshold': int,
            'batch_size': int,
            'epochs': int,
            'learning_rate': float,
            'device_name': str
        }
    }, ignore_extra_keys=True)
    config_schema.validate(settings)
    settings = settings['train']

    device = torch.device(settings['device_name'])
    print('Loading training data...')
    training_dataset = dataset_service.GanDataset(settings['training_data_folder'], settings['tile_filter_prefix'], settings['flow_threshold'])
    training_dataloader = DataLoader(training_dataset, batch_size=settings['batch_size'], shuffle=True)
    validation_dataset = dataset_service.GanDataset(settings['training_data_folder'], settings['tile_filter_prefix'], settings['flow_threshold'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    discriminator = discriminator_model.Discriminator(channels=1).to(device)
    generator = generator_model.Generator(in_channels=3, out_channels=1).to(device)

    disc_optimizer = optim.Adam(discriminator.parameters(), lr=settings['learning_rate'])
    gen_optimizer = optim.Adam(generator.parameters(), lr=settings['learning_rate'])
    bce_criterion = nn.BCELoss()

    gen_checkpoint_path = os.path.join(settings['checkpoint_folder'], 'gen.tar')
    disc_checkpoint_path = os.path.join(settings['checkpoint_folder'], 'disc.tar')

    print('Loading checkpoints')
    checkpoint_service.load_checkpoint(generator, gen_optimizer, device, settings['learning_rate'], gen_checkpoint_path)
    checkpoint_service.load_checkpoint(discriminator, disc_optimizer, device, settings['learning_rate'], disc_checkpoint_path)

    print(f"Training for {settings['epochs']} epochs")
    for index, epoch in enumerate(range(settings['epochs'])):
        train_service.train_gan(
            device, training_dataloader, discriminator, generator,
            disc_optimizer, gen_optimizer, bce_criterion
        )

        print('Saving checkpoints')
        checkpoint_service.save_checkpoint(generator, gen_optimizer, gen_checkpoint_path)
        checkpoint_service.save_checkpoint(discriminator, disc_optimizer, disc_checkpoint_path)

        print('Savings examples')
        checkpoint_service.save_example(generator, validation_dataloader, epoch, device, settings['examples_folder'])
