import os
from pathlib import Path
from rasterio import MemoryFile
from rasterio.windows import Window
import numpy as np
from .sketch_service import generate_sketch
import cv2


def is_subtile_valid(subtile, land_coverage_threshold):

    size = subtile.shape[0] * subtile.shape[1]
    sea = np.count_nonzero((subtile == 0) | (subtile == -9999))
    rate = (size-sea)/size
    return rate > land_coverage_threshold


def save_subtile(output_path, tile_id, dataset, x, y, size, land_coverage_threshold, flow_threshold, target_size):
    subtile_x = size * x
    subtile_y = size * y

    # Create a Window and calculate the transform from the source dataset
    window = Window(subtile_x, subtile_y, size, size)
    dem = dataset.read(1, window=window)

    if not is_subtile_valid(dem, land_coverage_threshold):
        return False
    
    # Prepare the sketch
    sketch = generate_sketch(dem, flow_threshold)
    sketch = np.transpose(sketch, (1, 2, 0)) # CHW -> HWC
    sketch = cv2.resize(sketch, (target_size, target_size))
    sketch = np.transpose(sketch, (2, 0, 1)) # HWC -> CHW

    # Prepare the dem
    dem = cv2.resize(dem, (target_size, target_size))
    dem = np.stack([dem], axis=0)
    
    # Save the 2 together as an numpy arrays file
    destination_path = os.path.join(output_path, tile_id) + f"_{x}_{y}.npz"
    np.savez(destination_path, dem=dem, sketch=sketch)

    return True

def save_tile(output_path, tile_id, tile_file_content, tile_size, subtile_size, land_coverage_threshold, flow_threshold, target_size):

    grid_count = tile_size//subtile_size

    skipped_subtiles = 0
    with MemoryFile() as memfile:
        memfile.write(tile_file_content)
        with memfile.open() as dataset:

            for i in range(grid_count):
                for j in range(grid_count):
                    saved = save_subtile(output_path, tile_id, dataset, i, j, subtile_size, land_coverage_threshold, flow_threshold, target_size)
                    if not saved:
                        skipped_subtiles += 1

            print(f'Skipped {skipped_subtiles} subtiles out of {grid_count**2} (below land cover threshold)')


def download_tile(download_url_root, output_path, tile_file_name, session, tile_size, subtile_size, land_coverage_threshold, flow_threshold, target_size): # If the file doesn't exist yet, download it

    tile_id = Path(tile_file_name).stem
    download_url = os.path.join(download_url_root, tile_file_name)


    flight_request = session.request('get', download_url) # Make a flight request to get a session cookie
    content_request = session.get(flight_request.url)
    if content_request.ok:
        save_tile(output_path, tile_id, content_request.content, tile_size, subtile_size, land_coverage_threshold, flow_threshold, target_size)
        print(f"Downloaded {tile_file_name}")
    else:
        print(f"Error downloading {download_url}: {content_request.status_code}, {content_request.text}")
