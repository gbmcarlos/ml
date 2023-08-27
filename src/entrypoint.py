import argparse
import lib.controllers as controllers
import yaml, logging, os


def run():

    args = setup()
    execute(args)

    return

def setup_arg_parser():

    parser = argparse.ArgumentParser(description="Prepare, train and export a Terrain Generation model")
    parser.add_argument('--settings-path', type=str, required=True)
    parser.add_argument('--results-path', type=str, required=True)
    subparsers = parser.add_subparsers(help='download-dems help', dest='tool')

    subparsers.add_parser('download', help='Download DEM files from a list, and split them into a grid of smaller DEMs')
    subparsers.add_parser('train', help='Train the cGAN model and export it to TorchScript')

    return parser

def setup_logging(results_path):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logpath = os.path.join(results_path, 'results.log')
    logger.addHandler(logging.FileHandler(logpath, '+a'))
    logging.print = logger.info

    os.makedirs(results_path, exist_ok=True)

def setup():

    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # setup_logging(args.results_path)

    return args

def read_settings(settings_path):
    try:
        with open(settings_path) as file:
            settings = yaml.load(file, Loader=yaml.FullLoader)
            return settings
    except Exception as e:
        print(f"An error occurred while reading the settings file: {e}")

def execute(args):
    
    settings = read_settings(args.settings_path)

    if args.tool == 'download':
        controllers.download(settings)
    elif args.tool == 'train':
        controllers.train(settings)

if __name__ == '__main__':
    run()
