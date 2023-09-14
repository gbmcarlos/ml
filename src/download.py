import os, requests
import argparse, yaml
import numpy as np
import cv2
from dotenv import load_dotenv
from schema import Schema, Optional

from lib.services import download_service, sketch_service


def run():

	parser = argparse.ArgumentParser(description="Download, split, and sketch DEMs")
	parser.add_argument('--settings-path', type=str, required=True)
	args = parser.parse_args()
	settings = read_settings(args.settings_path)

	for dem, tile_id in download_dems(settings):
		sketch_and_save(dem, tile_id, settings['flow_threshold'], settings['output_path'])

def read_settings(settings_path):
	try:
		with open(settings_path) as file:
			settings = yaml.load(file, Loader=yaml.FullLoader)
			config_schema = Schema({
				'download': {
					'credentials_env_file': str,
					'tiles_file': str,
					'url_root': str,
					'tile_size': int,
					'subtile_size': int,
					'target_size': int,
					'land_coverage_threshold': float,
					Optional('flow_threshold'): int,
					Optional('output_path'): str
				}
			}, ignore_extra_keys=True)
			config_schema.validate(settings)
			settings = settings['download']
			return settings
	except Exception as e:
		print(f"An error occurred while reading the settings file: {e}")

def sketch_and_save(dem, tile_id, flow_threshold, output_path):
	sketch = sketch_service.sketch_dem(dem, flow_threshold)
	sketch = np.transpose(sketch, (1, 2, 0)) # CHW -> HWC
	sketch = cv2.normalize(sketch, None, -1.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	# Prepare the dem
	dem = cv2.normalize(dem, None, -1.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	dem = np.transpose(np.stack([dem], axis=0), (1, 2, 0)) # HW -> HWC

	# Save the 2 together as an numpy arrays file
	destination_path = os.path.join(output_path, tile_id) + ".npz"
	np.savez(destination_path, dem=dem, sketch=sketch)

def download_dems(settings):

	if isinstance(settings, str):
		settings = read_settings(settings)

	load_dotenv(settings['credentials_env_file'])
	username = os.getenv("EARTHDATA_USERNAME")
	password = os.getenv("EARTHDATA_PASSWORD")

	tiles_file = open(settings['tiles_file'])  # The file containing a list of file names
	tile_file_names = tiles_file.read().splitlines()  # The list of file names

	downloader = download_service.DEMDownloader(
		settings['url_root'], username, password, settings['output_path'],
		settings['tile_size'], settings['subtile_size'], settings['target_size'],
		settings['land_coverage_threshold']
	)

	yield from downloader.download_tiles(tile_file_names)

if __name__ == '__main__':
	run()
