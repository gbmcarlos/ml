import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models import generator_model, discriminator_model
from ..services import dataset_service, train_service, checkpoint_service


def train_gan(training_data_file, batch_size, epochs, learning_rate, device_name, shuffle):

    device = torch.device(device_name)
    print('Loading training data...')
    training_dataset = dataset_service.GanDataset(training_data_file, device)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dataset = dataset_service.GanDataset(training_data_file, device)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    discriminator = discriminator_model.Discriminator(1).to(device)
    generator = generator_model.Generator(3, 1).to(device)

    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    bce_criterion = nn.BCELoss()

    print('Loading checkpoints')
    checkpoint_service.load_checkpoint(generator, gen_optimizer, device, learning_rate, 'data/gen.tar')
    checkpoint_service.load_checkpoint(discriminator, disc_optimizer, device, learning_rate, 'data/disc.tar')

    print(f"Training for {epochs} epochs")
    for index, epoch in enumerate(range(epochs)):
        train_service.train_gan(
            device, training_dataloader, discriminator, generator,
            disc_optimizer, gen_optimizer, bce_criterion
        )

        print('Saving checkpoints')
        checkpoint_service.save_checkpoint(generator, gen_optimizer, 'data/gen.tar')
        checkpoint_service.save_checkpoint(discriminator, disc_optimizer, 'data/disc.tar')

        print('Savings examples')
        checkpoint_service.save_example(generator, validation_dataloader, epoch, device, 'data/examples')
