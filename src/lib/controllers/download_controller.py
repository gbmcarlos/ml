import os
import requests
from dotenv import load_dotenv
import concurrent
from ..services import download_service
from schema import Schema


def download_dems(settings):

    config_schema = Schema({
        'download': {
            'credentials_env_file': str,
            'tiles_file': str,
            'url_root': str,
            'output_path': str,
            'parallel': bool,
            'tile_size': int,
            'subtile_size': int,
            'land_coverage_threshold': float
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

        print(f'Downloading {len(tile_file_names)} files to {settings["output_path"]} ' + ('parallelly' if settings["parallel"] else 'sequentially'))
        if settings['parallel']:
            with concurrent.futures.ProcessPoolExecutor() as process_pool:
                futures = (process_pool.submit(
                    download_service.download_tile,
                    settings['url_root'], 
                    settings['output_path'], 
                    tile_file_name.strip(), 
                    session, settings['tile_size'], 
                    settings['subtile_size'], 
                    settings['land_coverage_threshold']
                ) for tile_file_name in tile_file_names)

                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for tile_file_name in tile_file_names:
                download_service.download_tile(
                    settings['url_root'], 
                    settings['output_path'], 
                    tile_file_name.strip(), 
                    session, 
                    settings['tile_size'], 
                    settings['subtile_size'], 
                    settings['land_coverage_threshold']
                )

        print('Done!')
