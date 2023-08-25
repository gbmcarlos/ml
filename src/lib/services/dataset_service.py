import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .sketch_service import generate_sketch


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
        sketch, dem, tile_id = generate_sketch(self.data[index], self.flow_threshold, 256)
        sample = torch.from_numpy(sketch).float()
        sample = torch.permute(sample, (2, 0, 1))  # Channels first
        target = torch.from_numpy(dem).float()
        target = torch.unsqueeze(target, 0)

        return sample, target
