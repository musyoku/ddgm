import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from progress import Progress
from model import params_energy_model, params_generative_model, ddgm
from args import args
from dataset import load_rgb_images
from plot import plot

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x_batch = np.empty((batchsize, 3, height, width), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		x_batch[j] = images[data_index]
	return x_batch

def main():
	# load MNIST images
	images = load_rgb_images(args.image_dir)

	# config
	config_energy_model = to_object(params_energy_model["config"])
	config_generative_model = to_object(params_generative_model["config"])
	
	# settings
	max_epoch = 1000
	n_trains_per_epoch = 100
	batchsize_positive = 128
	batchsize_negative = 128
	plot_interval = 30

	# seed
	np.random.seed(args.seed)
	if args.gpu_enabled:
	    cuda.cupy.random.seed(args.seed)

	# init weightnorm layers
	if config_energy_model.use_weightnorm:
		print "initializing weight normalization layers of the energy model ..."
		x_positive = sample_from_data(images, batchsize_positive * 10)
		ddgm.compute_energy(x_positive)

	if config_generative_model.use_weightnorm:
		print "initializing weight normalization layers of the generative model ..."
		x_positive = sample_from_data(images, batchsize_positive * 10)
		ddgm.compute_energy(x_positive)

	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_energy_positive = 0
		sum_energy_negative = 0
		sum_kld = 0

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sample_from_data(images, batchsize_positive)

			# sample from generator
			x_negative = ddgm.generate_x(batchsize_negative)

			if True:
				loss, energy_positive, energy_negative = ddgm.compute_loss(x_positive, x_negative)
				ddgm.backprop_energy_model(loss)
			else:
				energy_positive, experts_positive = ddgm.compute_energy(x_positive)
				energy_positive = F.sum(energy_positive) / ddgm.get_batchsize(x_positive)
				ddgm.backprop_energy_model(energy_positive)
				
				energy_negative, experts_negative = ddgm.compute_energy(x_negative)
				energy_negative = F.sum(energy_negative) / ddgm.get_batchsize(x_negative)
				ddgm.backprop_energy_model(-energy_negative)

			# train generative model
			# TODO: KLD must be greater than or equal to 0
			x_negative = ddgm.generate_x(batchsize_negative)
			kld = ddgm.compute_kld_between_generator_and_energy_model(x_negative)
			ddgm.backprop_generative_model(kld)

			sum_energy_positive += float(energy_positive.data)
			sum_energy_negative += float(energy_negative.data)
			sum_kld += float(kld.data)
			if t % 10 == 0:
				progress.show(t, n_trains_per_epoch, {})

		progress.show(n_trains_per_epoch, n_trains_per_epoch, {
			"x+": int(sum_energy_positive / n_trains_per_epoch),
			"x-": int(sum_energy_negative / n_trains_per_epoch),
			"KLD": int(sum_kld / n_trains_per_epoch)
		})
		ddgm.save(args.model_dir)

		if epoch % plot_interval == 0 or epoch == 1:
			plot(filename="epoch_{}_time_{}min".format(epoch, progress.get_total_time()))

if __name__ == '__main__':
	main()
