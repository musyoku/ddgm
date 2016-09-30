import sys, os
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import params, ddgm

def main():
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_negative = ddgm.generate_x(100)
	if params.gpu_enabled:
		x_negative.to_cpu()
	visualizer.tile_x(x_negative.data, dir=args.plot_dir)

if __name__ == '__main__':
	main()
