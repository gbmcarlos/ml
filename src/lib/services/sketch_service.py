import numpy as np
import matplotlib.pyplot as plt
from pysheds.sview import Raster
from pysheds.grid import Grid
import pysheds.io as io
import cv2
from skimage.morphology import skeletonize


def sketch_dem(dem, flow_threshold, mode='stack'):  # Given a DEM, extract the sea, the rivers and the ridges, and put them together in a 3 channel image

	resized_dem = cv2.resize(dem, (128, 128))

	raster = Raster(resized_dem)
	grid = Grid.from_raster(raster)

	sea = extract_sea(raster)
	land_mask = (~sea.astype(bool)).astype(np.uint8)
	rivers = extract_rivers(grid, raster, land_mask, flow_threshold)
	ridges = extract_ridges(grid, raster, land_mask, flow_threshold)

	sea = cv2.resize(sea, (256, 256))
	rivers = simplify(cv2.resize(rivers, (256, 256)))
	ridges = simplify(cv2.resize(ridges, (256, 256)))

	if mode == 'stack':
		return np.stack([sea, rivers, ridges], axis=0)
	elif mode == 'mask':
		mask = np.zeros_like(sea) # Sea = 0
		mask[sea==0] = 2 # Land = 2
		mask[rivers==255] = 4 # Rivers = 4
		mask[ridges==255] = 8 # Ridges = 8
		mask = np.stack([mask], axis=0)
		return mask


def extract_sea(raster):  # Simple mask to extract the area of elevation 0
	sea = np.zeros_like(raster, dtype=np.uint8)
	sea[raster <= 0] = 255
	return sea


def extract_rivers(grid, raster, land_mask, flow_threshold):
	rivers = extract_flow(grid, raster, land_mask, flow_threshold)
	return rivers


def extract_ridges(grid, raster, land_mask, flow_threshold):  # Invert the DEM and calculate the same as rivers
	raster = raster.max() - raster
	ridges = extract_flow(grid, raster, land_mask, flow_threshold)
	return ridges


def extract_flow(grid, raster, land_mask, flow_threshold):  # Condition the DEM (resolving flats and pits), calculate flow direction and accumulation, normalize, threshold, and simplify
	conditioned_raster = condition_raster(grid, raster)

	direction_map = (64, 128, 1, 2, 4, 8, 16, 32)
	flow_direction = grid.flowdir(conditioned_raster, dirmap=direction_map)
	accumulation = grid.accumulation(flow_direction, dirmap=direction_map)

	flow = np.log(accumulation + 1)
	flow = (flow - np.amin(flow)) / (np.amax(flow) - np.amin(flow))
	flow = np.array(flow * 255, dtype=np.uint8) # Normalize to [0,255]

	_, flow = cv2.threshold(flow, flow_threshold, 255, cv2.THRESH_BINARY)
	flow *= land_mask

	return flow


def simplify(flow):  # Accentuate everything and skeletonize it to 1 pixel width

	output = flow
	output = cv2.dilate(output, kernel=np.ones((10, 10), np.uint8), iterations=1) # Accentuate
	output = (skeletonize(output)*255).astype('uint8') # Simplify

	return output


def condition_raster(grid, raster):
	pit_filled_raster = grid.fill_pits(raster)
	flooded_raster = grid.fill_depressions(pit_filled_raster)
	inflated_raster = grid.resolve_flats(flooded_raster)
	return inflated_raster
