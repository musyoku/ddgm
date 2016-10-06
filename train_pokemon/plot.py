import sys, os
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import params, dcdgm

def main():
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_negative = dcdgm.generate_x(100, test=True)
	if params.gpu_enabled:
		x_negative.to_cpu()
	rgba_image = np.ones((100, 4, params.x_height, params.x_width), dtype=np.float32)
	rgba_image[:,:3,:,:] = x_negative.data
	print x_negative.data.shape
	visualizer.tile_x(x_negative.data.transpose(0, 2, 3, 1), dir=args.plot_dir, image_width=params.x_width, image_height=params.x_height, image_channel=3)

if __name__ == '__main__':
	main()
