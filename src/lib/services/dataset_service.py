import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class GanDataset(Dataset):
    def __init__(self, data_dir, tile_filter_prefix, flow_threshold, transform=None):
        self.data_dir = data_dir
        path = os.path.join(data_dir, tile_filter_prefix + '*')
        self.data = glob.glob(path)
        self.flow_threshold = flow_threshold
        self.transform = transform
        print(f"Found {len(self.data)} data points")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.load(self.data[index])
        sample = torch.from_numpy(data['sketch']).float()
        target = torch.from_numpy(data['dem']).float()
        
        return sample, target
