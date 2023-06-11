import os
import glob
from pathlib import Path
import rasterio
from rasterio import MemoryFile
from rasterio.windows import Window
import numpy as np

def is_subtile_valid(subtile):

    threshold_rate = 0.2
    size = subtile.shape[0] * subtile.shape[1]
    coverage = np.count_nonzero((subtile != 0) & subtile != -9999)
    rate = coverage/size
    return rate > threshold_rate

def save_subtile(output_path, tile_id, dataset, x, y, size):
    subtile_x = size * x
    subtile_y = size * y


    # Create a Window and calculate the transform from the source dataset
    window = Window(subtile_x, subtile_y, size, size)
    subtile = dataset.read(1, window=window)

    if not is_subtile_valid(subtile):
        print(f'Skipping subtile ({x}, {y})')
        return

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

def save_tile(output_path, tile_id, tile_file_content):

    subtile_size = 720
    grid_count = 3600//720

    with MemoryFile() as memfile:
        memfile.write(tile_file_content)
        with memfile.open() as dataset:

            for i in range(grid_count):
                for j in range(grid_count):
                    save_subtile(output_path, tile_id, dataset, i, j, subtile_size)

def download_tile(download_url_root, output_path, tile_file_name, session): # If the file doesn't exist yet, download it

    tile_id = Path(tile_file_name).stem
    # Check if it's already there
    glob_path = os.path.join(output_path, tile_id) + "_\\d_\\d" + Path(tile_file_name).suffix
    result = glob.glob(fr'{glob_path}')
    print(result)
    if result:
        print(f"Files for {tile_file_name} already exist")
        return

    download_url = os.path.join(download_url_root, tile_file_name)

    flightRequest = session.request('get', download_url) # Make a flight request to get a session cookie
    contentRequest = session.get(flightRequest.url)
    if contentRequest.ok:
        save_tile(output_path, tile_id, contentRequest.content)
        print(f"Downloaded {tile_file_name}")
    else:
        print(f"Error downloading {download_url}: {contentRequest.status_code}, {contentRequest.text}")
