import argparse
import lib.controllers as controllers
import yaml

def run():

	args = setup()
	execute(args)

	return

def setup_arg_parser():

	parser = argparse.ArgumentParser(description="Prepare, train and export a Terrain Generation model")
	parser.add_argument('--settings-path', type=str, required=True)
	parser.add_argument('--name', type=str, required=False)
	subparsers = parser.add_subparsers(dest='tool')

	subparsers.add_parser('download', help='Download DEM files from a list, and split them into a grid of smaller DEMs')
	subparsers.add_parser('train', help='Train the cGAN model and export it to TorchScript')

	return parser

def setup():

	arg_parser = setup_arg_parser()
	args = arg_parser.parse_args()

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
	settings['name'] = args.name

	if args.tool == 'download':
		controllers.download(settings)
	elif args.tool == 'train':
		controllers.train(settings)

if __name__ == '__main__':
	run()
