import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from .sketch_service import generate_sketch


class GanDataset(Dataset):
	def __init__(self, data_dir, tile_filter_prefix, rebuild=False, flow_threshold=0):
		self.rebuild = rebuild
		self.flow_threshold = 0
		self.data_dir = data_dir
		path = os.path.join(data_dir, tile_filter_prefix + '*')
		self.data = glob.glob(path)
		print(f"Found {len(self.data)} training samples")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):

		data = np.load(self.data[index])

		dem_raw = data['dem']
		dem = torch.from_numpy(dem_raw)
		dem = torch.permute(dem, (2, 0, 1)) # HWC -> CHW

		if (self.rebuild):
			dem_raw = np.squeeze(dem_raw, axis=2) # CHW -> HWC
			sketch_raw = generate_sketch(dem_raw, self.flow_threshold)
			sketch_raw = np.transpose(sketch_raw, (1, 2, 0)) # CHW -> HWC
			sketch_raw = cv2.normalize(sketch_raw, None, -1.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		else:
			sketch_raw = data['sketch']
		
		sketch = torch.from_numpy(sketch_raw)
		sketch = torch.permute(sketch, (2, 0, 1)) # HWC -> CHW
		noise = torch.rand((1, sketch.shape[1], sketch.shape[2]))
		sketch = torch.cat([noise, sketch], dim=0) # Add a channel of noise
		
		return sketch, dem
