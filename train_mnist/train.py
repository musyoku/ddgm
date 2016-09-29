from scipy.io import wavfile
import numpy as np
import os, sys, time
from chainer import cuda, Variable
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from model import params, ddgm
from args import args
from dataset import binarize_data, load_images
import dataset

def sample_from_data(images, batchsize, ndim_x):
	x_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		img = images[data_index]
		x_batch[j] = img.reshape((ndim_x,))

	# binalize
	x_batch = binarize_data(x_batch)

	return x_batch

def main():
	# load MNIST images
	images = load_images(args.train_image_dir)
	
	# settings
	max_epoch = 1000
	n_trains_per_epoch = 300
	batchsize = 100

	# seed
	np.random.seed(args.seed)
	if params.gpu_enabled:
	    cuda.cupy.random.seed(args.seed)

	total_time = 0
	for epoch in xrange(1, max_epoch):
		sum_loss = 0
		epoch_time = time.time()

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sample_from_data(images, batchsize, params.ndim_x)
			# sample from generator
			x_negative = ddgm.generate_x(batchsize)

			# train
			loss = ddgm.compute_loss(x_positive, x_negative)
			ddgm.backprop(loss)

			sum_loss += float(loss.data)
			if t % 10 == 0:
				sys.stdout.write("\rTraining in progress...({} / {})".format(t, n_trains_per_epoch))
				sys.stdout.flush()

		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch: {} loss: {:.3f} time: {} min total: {} min".format(epoch + 1, sum_loss / n_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
		sys.stdout.flush()
		ddgm.save(args.model_dir)

if __name__ == '__main__':
	main()
