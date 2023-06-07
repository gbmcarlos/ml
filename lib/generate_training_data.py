import os
import numpy as np
import matplotlib.pyplot as plt
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
import rasterio
import cv2
import skimage
import tempfile
import stat

def generate_sketch(input_file_path, options):
    sketch_id = os.path.basename(input_file_path)
    
    grid, dem, original_dem = preprocess_dem(input_file_path, options)
    
    sea = extract_sea(grid, dem)
    land_mask = (~sea.astype(bool)).astype(np.uint8)
    rivers = extract_rivers(grid, dem, land_mask, options)
    ridges = extract_ridges(grid, dem, land_mask, options)

    sketch = cv2.merge([sea, rivers, ridges]) # BGR

    return (sketch_id, original_dem, sketch)

def preprocess_dem(input_file_path, options):
    grid = Grid.from_raster(input_file_path)
    original_raster = rasterio.open(input_file_path).read(1)

    blurred_dem = blur(original_raster, options["dem_blurring_iterations"])

    new_view_finder = ViewFinder(
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
        shape=blurred_dem.shape
    )
    new_raster = Raster(blurred_dem, new_view_finder)
    grid.viewfinder = new_view_finder
    return grid, new_raster, original_raster

def extract_sea(grid, dem):
    sea = np.zeros_like(dem, dtype=np.uint8)
    sea[dem <= 0] = 255
    return sea

def extract_rivers(grid, dem, land_mask, options):
    rivers = extract_flow(grid, dem, land_mask, options)
    return rivers


def extract_ridges(grid, dem, land_mask, options):
    dem = dem.max() - dem

    ridges = extract_flow(grid, dem, land_mask, options)
    return ridges

def extract_flow(grid, dem, land_mask, options):
    conditioned_dem = condition_dem(grid, dem)

    direction_map = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_direction = grid.flowdir(conditioned_dem, dirmap=direction_map)
    accumulation = grid.accumulation(flow_direction, dirmap=direction_map)

    flow = accumulation
    flow = np.log(accumulation + 1)
    flow = (flow - np.amin(flow)) / (np.amax(flow) - np.amin(flow))
    flow = np.array(flow * 255, dtype=np.uint8) # Normalize to [0,255]

    _, flow = cv2.threshold(flow, options["flow_threshold"], 255, cv2.THRESH_BINARY)
    flow *= land_mask

    # flow = skimage.morphology.skeletonize(flow).astype(np.uint8) * 255
    # flow = smoothen(flow, 7, 16, -4)

    return flow

def smoothen(data, blur, sharp1, sharp2):
    blurred = cv2.GaussianBlur(data, (blur, blur), 0)
    unsharp_mask = cv2.subtract(data, blurred)
    sharp = cv2.addWeighted(data, sharp1, unsharp_mask, sharp2, 0)
    return sharp

def condition_dem(grid, dem):
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    return inflated_dem

def blur(data, iterations):
    for i in range(iterations):
        data = downsample(data)
    for i in range(iterations):
        data = upsample(data)
    return data

def upsample(data):
    return cv2.pyrUp(data, dstsize=(
        data.shape[1] * 2,
        data.shape[0] * 2
    ))

def downsample(data):
    return cv2.pyrDown(data, dstsize=(
        data.shape[1] // 2,
        data.shape[0] // 2
    ))