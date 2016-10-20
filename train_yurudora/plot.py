import sys, os
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import params, dcdgm
from dataset import binarize_data, load_rgba_images

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x_batch = np.full((batchsize, 3, height, width), -1,  dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		image_rgba = images[data_index]
		mask = np.repeat(image_rgba[3].reshape((-1, height, width)), 3, axis=0)
		image_rgb = image_rgba[:3] * mask
		x_batch[j] *= 1 - mask
		x_batch[j] += image_rgb
	return x_batch

def main():
	# load MNIST images
	images = load_rgba_images(args.image_dir)
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_negative = dcdgm.generate_x(100, test=True)
	if params.gpu_enabled:
		x_negative.to_cpu()
	x_negative = x_negative.data

	x_negative = sample_from_data(images, 100)


	rgba_image = np.ones((100, 4, params.x_height, params.x_width), dtype=np.float32)
	# rgba_image[:,:3,:,:] = np.clip((x_negative + 1) / 2, 0, 1)
	x_negative[x_negative < 0] = 1
	x_negative = np.clip(x_negative, 0, 1)
	visualizer.tile_x(x_negative.transpose(0, 2, 3, 1), dir=args.plot_dir, image_width=params.x_width, image_height=params.x_height, image_channel=3)

if __name__ == '__main__':
	main()
