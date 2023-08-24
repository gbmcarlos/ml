import torch

def train_gan(device, dataloader, discriminator, generator, disc_optimizer, gen_optimizer, loss_fn):

    generator.eval()
    discriminator.eval()

    for _, (sample, target) in enumerate(dataloader):

        sample = sample.to(device)
        target = target.to(device)

        # Train the discriminator
        discriminator.train()
        disc_optimizer.zero_grad()

        disc_pred_real = discriminator(target)
        disc_loss_real = loss_fn(disc_pred_real, torch.ones_like(disc_pred_real))
        disc_pred_fake = discriminator(generator(sample))
        disc_loss_fake = loss_fn(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        disc_total_loss = disc_loss_fake + disc_loss_real
        
        disc_total_loss.backward()
        disc_optimizer.step()
        discriminator.eval()

        # Train the generator
        generator.train()
        gen_optimizer.zero_grad()
        
        disc_pred_fake = discriminator(generator(sample))
        disc_loss_fake = loss_fn(disc_pred_fake, torch.ones_like(disc_pred_fake))
        
        disc_loss_fake.backward()
        gen_optimizer.step()
        generator.eval()
