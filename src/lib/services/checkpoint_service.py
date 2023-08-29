import os
import torch


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
