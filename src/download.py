import os, requests
import argparse, yaml
from dotenv import load_dotenv
from schema import Schema, Optional

from lib.services import download_service


def run():

	parser = argparse.ArgumentParser(description="Download, split, and sketch DEMs")
	parser.add_argument('--settings-path', type=str, required=True)
	args = parser.parse_args()
	settings = read_settings(args.settings_path)

	download_dems(settings)

	return

def read_settings(settings_path):
	try:
		with open(settings_path) as file:
			settings = yaml.load(file, Loader=yaml.FullLoader)
			return settings
	except Exception as e:
		print(f"An error occurred while reading the settings file: {e}")

def download_dems(settings):

	config_schema = Schema({
		'download': {
			'credentials_env_file': str,
			'tiles_file': str,
			'url_root': str,
			'output_path': str,
			'tile_size': int,
			'subtile_size': int,
			'land_coverage_threshold': float,
			'flow_threshold': int,
			'target_size': int
		}
	}, ignore_extra_keys=True)
	config_schema.validate(settings)
	settings = settings['download']

	load_dotenv(settings['credentials_env_file'])
	username = os.getenv("EARTHDATA_USERNAME")
	password = os.getenv("EARTHDATA_PASSWORD")

	tiles_file = open(settings['tiles_file'])  # The file containing a list of file names
	tile_file_names = tiles_file.readlines()  # The list of file names

	os.makedirs(settings['output_path'], exist_ok=True)

	with requests.Session() as session:
		session.auth = (username, password)

		print(f'Processing {len(tile_file_names)} tiles to {settings["output_path"]}')
		for tile_file_name in tile_file_names:
			download_service.download_tile(
				settings['url_root'], 
				settings['output_path'], 
				tile_file_name.strip(), 
				session, 
				settings['tile_size'], 
				settings['subtile_size'], 
				settings['land_coverage_threshold'],
				settings['flow_threshold'], 
				settings['target_size']
			)

		print('Done!')

if __name__ == '__main__':
	run()
