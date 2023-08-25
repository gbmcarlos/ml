import os
import numpy as np
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import cv2
from skimage.morphology import skeletonize


def generate_sketch(input_file_path, flow_threshold, target_size):  # Given a DEM, extract the sea, the rivers and the ridges, and put them together in a 3 channel image
    tile_id = os.path.basename(input_file_path)

    grid = Grid.from_raster(input_file_path)
    dem = grid.read_raster(input_file_path)

    sea = extract_sea(dem)
    land_mask = (~sea.astype(bool)).astype(np.uint8)
    rivers = extract_rivers(grid, dem, land_mask, flow_threshold)
    ridges = extract_ridges(grid, dem, land_mask, flow_threshold)

    sketch = cv2.merge([sea, rivers, ridges])  # BGR
    sketch = cv2.resize(sketch, (target_size, target_size))
    dem = cv2.resize(dem, (target_size, target_size))

    print(f"Sketched {tile_id}")

    return sketch, dem, tile_id


def extract_sea(dem):  # Simple mask to extract the area of elevation 0
    sea = np.zeros_like(dem, dtype=np.uint8)
    sea[dem <= 0] = 255
    return sea


def extract_rivers(grid, dem, land_mask, flow_threshold):
    rivers = extract_flow(grid, dem, land_mask, flow_threshold)
    return rivers


def extract_ridges(grid, dem, land_mask, flow_threshold):  # Invert the DEM and calculate the same as rivers
    dem = dem.max() - dem
    ridges = extract_flow(grid, dem, land_mask, flow_threshold)
    return ridges


def extract_flow(grid, dem, land_mask, flow_threshold):  # Condition the DEM (resolving flats and pits), calculate flow direction and accumulation, normalize, threshold, and simplify
    conditioned_dem = condition_dem(grid, dem)

    direction_map = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_direction = grid.flowdir(conditioned_dem, dirmap=direction_map)
    accumulation = grid.accumulation(flow_direction, dirmap=direction_map)

    flow = np.log(accumulation + 1)
    flow = (flow - np.amin(flow)) / (np.amax(flow) - np.amin(flow))
    flow = np.array(flow * 255, dtype=np.uint8) # Normalize to [0,255]

    _, flow = cv2.threshold(flow, flow_threshold, 255, cv2.THRESH_BINARY)
    flow = simplify(flow)
    flow *= land_mask

    return flow


def simplify(flow):  # Accentuate everything and skeletonize it to 1 pixel width

    output = flow
    output = cv2.dilate(output, kernel=np.ones((10, 10), np.uint8), iterations=1) # Accentuate
    output = (skeletonize(output)*225).astype('uint8') # Simplify

    return output


def condition_dem(grid, dem):
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    return inflated_dem


def plot(title, image):
    print(title)
    plt.imshow(image, cmap='Greys')
    plt.show()