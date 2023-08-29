import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class GanDataset(Dataset):
    def __init__(self, data_dir, tile_filter_prefix, sample_transform=None, target_transform=None):
        self.data_dir = data_dir
        path = os.path.join(data_dir, tile_filter_prefix + '*')
        self.data = glob.glob(path)
        print(f"Found {len(self.data)} training samples")
        self.sample_transform = sample_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.load(self.data[index])
        sample = torch.permute(torch.from_numpy(data['sketch']), (2, 0, 1))
        target = torch.permute(torch.from_numpy(data['dem']), (2, 0, 1))

        if self.sample_transform:
            sample = self.sample_transform(sample)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sample, target
