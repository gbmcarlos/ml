import pathlib
import os
import concurrent
import numpy as np
from ..services import sketch_service
from schema import Schema
from PIL import Image


def sketch_dem(settings):

    config_schema = Schema({
        'sketch': {
            'input_path': str,
            'tile_filter_prefix': str,
            'output_folder': str,
            'flow_threshold': int,
            'target_size': int,
            'parallel': bool
        }
    }, ignore_extra_keys=True)
    config_schema.validate(settings)
    settings = settings['sketch']

    tile_file_paths = [f for f in pathlib.Path(settings['input_path']).glob(settings['tile_filter_prefix'] + '*')]

    print(f'Sketching {len(tile_file_paths)} DEMs to {settings["output_folder"]} ' + ('parallelly' if settings["parallel"] else 'sequentially'))
    if not settings['parallel']:
        for i, tile_file_path in enumerate(tile_file_paths):
            sketch, dem, tile_id = sketch_service.generate_sketch(
                str(tile_file_path), 
                settings['flow_threshold'],
                settings['target_size']
            )
            image = Image.fromarray(sketch)
            image.save(os.path.join(settings['output_folder'], tile_id))

    else:
        with concurrent.futures.ThreadPoolExecutor() as process_pool:
            futures = (process_pool.submit(
                sketch_service.generate_sketch,
                str(tile_file_path), settings['flow_threshold'],
            ) for i, tile_file_path in enumerate(tile_file_paths))

            for future in concurrent.futures.as_completed(futures):
                sketch, dem, tile_id = future.result()
                image = Image.fromarray(sketch)
                image.save(os.path.join(settings['output_folder'], tile_id))

    print('Done!')
