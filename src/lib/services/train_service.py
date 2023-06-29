import torch
from tqdm import tqdm

def train_gan(device, dataloader, discriminator, generator, disc_optimizer, gen_optimizer, bce_criterion):

    correct_classifications = 0
    wrong_classifications = 0

    loop = tqdm(dataloader)
    for step, (sample, target) in enumerate(loop):

        sample = sample.to(device)
        target = target.to(device)

        fake_sample = generator(sample)
        disc_fake = discriminator(fake_sample)
        disc_real = discriminator(target)

        # Train the discriminator
        disc_loss_fake = bce_criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss_real = bce_criterion(disc_real, torch.ones_like(disc_real))
        disc_total_loss = (disc_loss_fake + disc_loss_real)

        discriminator.zero_grad()
        disc_total_loss.backward(retain_graph=True)
        disc_optimizer.step()

        # Train the generator
        gen_fake_loss = bce_criterion(fake_sample, torch.ones_like(fake_sample))
        gen_optimizer.zero_grad()
        gen_fake_loss.backward()
        gen_optimizer.step()

        classification = torch.sigmoid(disc_fake).mean().item()
        if classification > 0.5:  # If the discriminator classifies it as real
            wrong_classifications += 1
        else:
            correct_classifications += 1
        disc_score = correct_classifications / (correct_classifications + wrong_classifications)

        loop.set_postfix(
            disc_score=disc_score
        )
