import pathlib
import os
import concurrent
import numpy as np
from ..services import sketch_service


def sketch_dem(parallel, input_path, flow_threshold, output_file):
    dems = []
    sketches = []
    tile_file_paths = [f for f in pathlib.Path(input_path).glob('*.tif')]

    print(f'Sketching {len(tile_file_paths)} DEMs to {output_file} ' + ('parallelly' if parallel else 'sequentially'))
    for i, tile_file_path in enumerate(tile_file_paths):
        tile_file_path = str(tile_file_path)

        if not parallel:
            sketch_id, dem, sketch = sketch_service.generate_sketch(tile_file_path, flow_threshold)
            dems.append(dem)
            sketches.append(sketch)

        else:
            with concurrent.futures.ProcessPoolExecutor() as process_pool:
                futures = []
                for tile_file_path in tile_file_paths:
                    tile_file_path = str(tile_file_path)
                    futures.append(process_pool.submit(sketch_service.generate_sketch, tile_file_path, flow_threshold))

    if parallel:
        for future in concurrent.futures.as_completed(futures):
            future.result()

    training_input = np.array(sketches)
    training_output = np.array(dems)
    np.savez(output_file, x=training_input, y=training_output)

    print('Done!')
