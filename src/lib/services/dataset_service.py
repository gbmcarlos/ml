import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class GanDataset(Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.data = os.listdir(data_dir)
		print(f"Found {len(self.data)} training samples")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):

		data = np.load(os.path.join(self.data_dir, self.data[index]))
		dem = torch.from_numpy(data['dem'])
		
		sketch = torch.from_numpy(data['sketch'])
		noise = torch.rand((1, sketch.shape[1], sketch.shape[2]))
		sketch = torch.cat([noise, sketch], dim=0) # Add a channel of noise
		
		return sketch, dem
