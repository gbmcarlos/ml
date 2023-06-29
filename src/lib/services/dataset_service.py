import numpy as np
import torch
from torch.utils.data import Dataset


class GanDataset(Dataset):
    def __init__(self, data_file, device):
        dataset = np.load('data/training_data.npz')
        self.samples = dataset['x']
        self.targets = dataset['y']
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_image = self.samples[index]
        sample = torch.from_numpy(input_image).float()
        sample = torch.permute(sample, (2, 0, 1))  # Channels first

        output_image = self.targets[index]
        target = torch.from_numpy(output_image).float()

        return sample, target
