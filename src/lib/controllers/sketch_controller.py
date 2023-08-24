import pathlib
import os
import concurrent
import numpy as np
from ..services import sketch_service
from schema import Schema

def sketch_dem(settings):

    config_schema = Schema({
        'sketch': {
            'input_path': str,
            'tile_filter_prefix': str,
            'output_file': str,
            'flow_threshold': str,
            'parallel': bool
        }
    }, ignore_extra_keys=True)
    config_schema.validate(settings)
    settings = settings['sketch']

    dems = []
    sketches = []
    tile_file_paths = [f for f in pathlib.Path(settings['input_path']).glob(settings['tile_filter_prefix'] + '*')]

    print(f'Sketching {len(tile_file_paths)} DEMs to {settings["output_file"]} ' + ('parallelly' if settings["parallel"] else 'sequentially'))
    if not settings['parallel']:
        for i, tile_file_path in enumerate(tile_file_paths):
            sketch, dem = sketch_service.generate_sketch(
                str(tile_file_path), 
                settings['flow_threshold']
            )
            sketches.append(sketch)
            dems.append(dem)

    else:
        with concurrent.futures.ThreadPoolExecutor() as process_pool:
            futures = (process_pool.submit(
                sketch_service.generate_sketch,
                str(tile_file_path), settings['flow_threshold'],
            ) for i, tile_file_path in enumerate(tile_file_paths))

            for future in concurrent.futures.as_completed(futures):
                sketch, dem = future.result()
                sketches.append(sketch)
                dems.append(dem)

    training_input = np.array(sketches)
    training_output = np.array(dems)

    print('Saving training data file')
    np.savez(settings['output_file'], x=training_input, y=training_output)

    print('Done!')
