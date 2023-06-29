import os
import torch
from torchvision.utils import save_image


def save_checkpoint(model, optimizer, output_file):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, output_file)

def load_checkpoint(model, optimizer, device, learning_rate, input_file):
    if os.path.exists(input_file):
        checkpoint = torch.load(input_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

def save_example(generator, dataloader, epoch, device, output_folder):

    sample, target = next(iter(dataloader))
    sample, target = sample.to(device), target.to(device)

    generator.eval()
    with torch.no_grad():
        fake_target = generator(sample)
        # fake_target = fake_target * 0.5 + 0.5  # remove normalization#
        save_image(sample, output_folder + f"/sample_{epoch}.png")
        save_image(fake_target, output_folder + f"/generated_{epoch}.png")
    generator.train()
