import os
from pathlib import Path
import rasterio
from rasterio import MemoryFile
from rasterio.windows import Window
import numpy as np


def is_subtile_valid(subtile, land_coverage_threshold):

    size = subtile.shape[0] * subtile.shape[1]
    sea = np.count_nonzero((subtile == 0) | (subtile == -9999))
    rate = (size-sea)/size
    return rate > land_coverage_threshold


def save_subtile(output_path, tile_id, dataset, x, y, size, land_coverage_threshold):
    subtile_x = size * x
    subtile_y = size * y

    # Create a Window and calculate the transform from the source dataset
    window = Window(subtile_x, subtile_y, size, size)
    subtile = dataset.read(1, window=window)


    if not is_subtile_valid(subtile, land_coverage_threshold):
        return False

    transform = dataset.window_transform(window)

    # Create a new cropped raster to write to
    profile = dataset.profile
    profile.update({
        'height': size,
        'width': size,
        'transform': transform,
        'nodata': -9999
    })

    destination_path = os.path.join(output_path, tile_id) + f"_{x}_{y}.tif"

    with rasterio.open(destination_path, 'w', **profile) as dst:
        # Read the data from the window and write it to the output raster
        dst.write(subtile, 1)

    return True


def save_tile(output_path, tile_id, tile_file_content, tile_size, subtile_size, land_coverage_threshold):

    grid_count = tile_size//subtile_size

    skipped_subtiles = 0
    with MemoryFile() as memfile:
        memfile.write(tile_file_content)
        with memfile.open() as dataset:

            for i in range(grid_count):
                for j in range(grid_count):
                    saved = save_subtile(output_path, tile_id, dataset, i, j, subtile_size, land_coverage_threshold)
                    if not saved:
                        skipped_subtiles += 1

            print(f'Skipped {skipped_subtiles} subtiles out of {grid_count**2} (below land cover threshold)')


def download_tile(download_url_root, output_path, tile_file_name, session, tile_size, subtile_size, land_coverage_threshold): # If the file doesn't exist yet, download it

    tile_id = Path(tile_file_name).stem
    download_url = os.path.join(download_url_root, tile_file_name)

    flight_request = session.request('get', download_url) # Make a flight request to get a session cookie
    content_request = session.get(flight_request.url)
    if content_request.ok:
        save_tile(output_path, tile_id, content_request.content, tile_size, subtile_size, land_coverage_threshold)
        print(f"Downloaded {tile_file_name}")
    else:
        print(f"Error downloading {download_url}: {content_request.status_code}, {content_request.text}")
