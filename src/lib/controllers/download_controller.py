import os
import requests
from dotenv import load_dotenv
import concurrent
from ..services import download_service


def download_dems(parallel, credentials_env_file, tiles_file, url_root, output_path, tile_size, subtile_size, land_coverage_threshold):

    load_dotenv(credentials_env_file)
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    tiles_file = open(tiles_file)  # The file containing a list of file names
    tile_file_names = tiles_file.readlines()  # The list of file names

    with requests.Session() as session:
        session.auth = (username, password)

        print(f'Downloading {len(tile_file_names)} files to {output_path} ' + ('parallelly' if parallel else 'sequentially'))
        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as process_pool:
                futures = (process_pool.submit(
                    download_service.download_tile,
                    url_root, output_path, tile_file_name.strip(), session, tile_size, subtile_size, land_coverage_threshold
                ) for tile_file_name in tile_file_names)

                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for tile_file_name in tile_file_names:
                download_service.download_tile(
                    url_root, output_path, tile_file_name.strip(), session, tile_size, subtile_size, land_coverage_threshold
                )

        print('Done!')
