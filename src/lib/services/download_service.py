import os, requests
from pathlib import Path
from rasterio import MemoryFile
from rasterio.windows import Window
import numpy as np
import cv2


class DEMDownloader():
	def __init__(self, url_root, username, password, output_path, tile_size, subtile_size, target_size, land_coverage_threshold):
		self.url_root = url_root
		with requests.Session() as session:
			self.session = session
			self.session.auth = (username, password)
		self.username = username
		self.password = password
		self.output_path = output_path
		os.makedirs(output_path, exist_ok=True)
		self.subtile_size = subtile_size
		self.grid_count = tile_size//subtile_size
		self.target_size = target_size
		self.land_coverage_threshold = land_coverage_threshold

	def download_tiles(self, tile_file_names):
		for tile_file_name in tile_file_names:
			tile_id = Path(tile_file_name).stem
			print(f"Downloading {tile_id} ...")
			
			download_url = os.path.join(self.url_root, tile_file_name)
			flight_request = self.session.request('get', download_url) # Make a flight request to get a session cookie
			content_request = self.session.get(flight_request.url)
			
			if content_request.ok:
				yield from self.download_tile(tile_id, content_request.content)
			else:
				print(f"Error downloading {download_url}: {content_request.status_code}, {content_request.text}")

	def download_tile(self, tile_id, tile_data):

		with MemoryFile() as memfile:
			memfile.write(tile_data)
			with memfile.open() as dataset:

				for i in range(self.grid_count):
					for j in range(self.grid_count):
						subtile = self.get_subtile(dataset, i, j)
						if subtile is not None:
							yield subtile, tile_id + f"_{i}_{j}"

	def get_subtile(self, dataset, x, y):
		subtile_x = self.subtile_size * x
		subtile_y = self.subtile_size * y

		# Create a Window and calculate the transform from the source dataset
		window = Window(subtile_x, subtile_y, self.subtile_size, self.subtile_size)
		dem = dataset.read(1, window=window)

		if not self.is_subtile_valid(dem):
			return None
		
		# Prepare the sketch
		dem = cv2.resize(dem, (self.target_size, self.target_size))
		return dem

	def is_subtile_valid(self, subtile):

		size = subtile.shape[0] * subtile.shape[1]
		sea = np.count_nonzero((subtile == 0) | (subtile == -9999))
		rate = (size-sea)/size
		return rate > self.land_coverage_threshold
