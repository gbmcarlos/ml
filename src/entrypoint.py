import argparse
import lib.controllers as controllers


def run():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="Prepare, train and export a Terrain Generation model")
    subparsers = parser.add_subparsers(help='download-dems help', dest='tool')

    parser_download = subparsers.add_parser('download', help='Download DEM files from a list, and split them into a grid of smalled DEMs')
    parser_download.add_argument("-p", "--parallel", action='store_true', help='Whether to execute in parallel (with multiple processes), instead of sequentially. Defaults to False')
    parser_download.add_argument("-c", "--credentials-env-file", default='.env', help='File to read, as and env vars file, to extract the credentials for Earthdata')
    parser_download.add_argument("-t", "--tiles-file", required=True, help='Path to a file containing a list of DEM file names to download from Earthdata')
    parser_download.add_argument("-u", "--url-root", required=True, help='Base url to use as prefix for every DEM file')
    parser_download.add_argument("-o", "--output-path", required=True, help='Output path where the result DEM files will be saved')
    parser_download.add_argument("-s", "--tile-size", type=int, default=720, help='Square size of the downloaded DEM tiles')
    parser_download.add_argument("-b", "--subtile-size", type=int, default=720, help='Square size to use to split the downloaded DEM into a grid of subtiles')
    parser_download.add_argument("-l", "--land-coverage-threshold", type=float, default=0.2, help='Rate (between 0 and 1) to use as threshold to determine whether a subtile contains enough land to be considered valid')

    parser_sketch = subparsers.add_parser('sketch', help='Generate a high-level sketch of each DEM, and save the resulting collection of pairs')
    parser_sketch.add_argument("-p", "--parallel", action='store_true', help='Whether to execute in parallel (with multiple processes), instead of sequentially. Defaults to False')
    parser_sketch.add_argument("-i", "--input-path", required=True, help='Path where the input DEM can be found')
    parser_sketch.add_argument("-f", "--flow-threshold", default=230, type=int, help='Binary threshold to use when computing flows (rivers and ridges)')
    parser_sketch.add_argument("-o", "--output-file", required=True, help='Path and name of the file to save the pairs of DEMs and generated sketches, as a NPZ file')

    parser_train = subparsers.add_parser('train', help='Train the cGAN model and export it to TorchScript')
    parser_train.add_argument("-t", "--training-data-file", required=True, help='Path where the training data file can be found, to be loaded with numpy')
    parser_train.add_argument("-b", "--batch-size", type=int, default=1, help='Number of samples for each batch')
    parser_train.add_argument("-e", "--epochs", type=int, default=1, help='Number of epochs to train for')
    parser_train.add_argument("-l", "--learning-rate", type=float, default=1e-2, help='Learning rate')
    parser_train.add_argument("-d", "--device-name", default='cpu', help="pyTorch device name to use")
    parser_train.add_argument("-s", "--shuffle", action='store_true', help='Whether to shuffle the samples when using the dataloader')

    args = parser.parse_args()
    tool = args.tool
    args = vars(args)
    args.pop('tool')

    if tool == 'download':
        controllers.download(**args)
    elif tool == 'sketch':
        controllers.sketch(**args)
    elif tool == 'train':
        controllers.train_gan(**args)


if __name__ == '__main__':
    run()
