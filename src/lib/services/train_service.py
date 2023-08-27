import torch
import wandb

def train_gan(device, dataloader, discriminator, generator, disc_optimizer, gen_optimizer, loss_fn, log_example_frequency):

    wandb.init(
        project='terramorph',
        config={

        }
    )

    generator.eval()
    discriminator.eval()

    for step, (sample, target) in enumerate(dataloader):

        sample = sample.to(device) # sketch
        target = target.to(device) # real dem

        # Train the discriminator
        discriminator.train()
        disc_optimizer.zero_grad()

        disc_pred_real = discriminator(sample, target)
        disc_loss_real = loss_fn(disc_pred_real, torch.ones_like(disc_pred_real))
        disc_pred_fake = discriminator(sample, generator(sample))
        disc_loss_fake = loss_fn(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        disc_total_loss = disc_loss_fake + disc_loss_real

        wandb.log({
            'disc_loss_fake': disc_loss_fake,
            'disc_loss_real': disc_loss_real
        })
        
        disc_total_loss.backward()
        disc_optimizer.step()
        discriminator.eval()

        # Train the generator
        generator.train()
        gen_optimizer.zero_grad()
        
        disc_pred_fake = discriminator(sample, generator(sample))
        gen_loss = loss_fn(disc_pred_fake, torch.ones_like(disc_pred_fake))

        wandb.log({
            'gen_loss': gen_loss
        })
        
        gen_loss.backward()
        gen_optimizer.step()
        generator.eval()

        if (step % log_example_frequency == 0):
            visualize_kernels(generator)

def visualize_kernels(model):
    initial_block_kernels = model.initial_block[0].weight.detach().clone()
    print(initial_block_kernels.shape)