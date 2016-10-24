import numpy as np
import os, sys, time, random, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from progress import Progress
from model import params_energy_model, params_generative_model, ddgm
from args import args
import sampler

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

def main():
	# settings
	max_epoch = 1000
	n_trains_per_epoch = 500
	batchsize_positive = 100
	batchsize_negative = 100

	# config
	config_energy_model = to_object(params_energy_model["config"])
	config_generative_model = to_object(params_generative_model["config"])

	# seed
	np.random.seed(args.seed)
	if args.gpu_enabled:
	    cuda.cupy.random.seed(args.seed)

	# init weightnorm layers
	if config_energy_model.use_weightnorm:
		print "initializing weight normalization layers of the energy model ..."
		x_positive = sampler.sample_from_gaussian_mixture(batchsize_positive * 10, 2, 10)
		ddgm.compute_energy(x_positive)

	if config_generative_model.use_weightnorm:
		print "initializing weight normalization layers of the generative model ..."
		x_negative = ddgm.generate_x(batchsize_negative * 10)

	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_energy_positive = 0
		sum_energy_negative = 0
		sum_kld = 0

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sampler.sample_from_gaussian_mixture(batchsize_positive, 2, 10)

			# sample from generator
			x_negative = ddgm.generate_x(batchsize_negative)

			# train energy model
			energy_positive = ddgm.compute_energy_sum(x_positive)
			energy_negative = ddgm.compute_energy_sum(x_negative)
			loss = energy_positive - energy_negative
			ddgm.backprop_energy_model(loss)
			
			# train generative model
			x_negative = ddgm.generate_x(batchsize_negative)
			kld = ddgm.compute_kld_between_generator_and_energy_model(x_negative)
			ddgm.backprop_generative_model(kld)

			sum_energy_positive += float(energy_positive.data)
			sum_energy_negative += float(energy_negative.data)
			sum_kld += float(kld.data)
			if t % 10 == 0:
				progress.show(t, n_trains_per_epoch, {})

		progress.show(n_trains_per_epoch, n_trains_per_epoch, {
			"x+": sum_energy_positive / n_trains_per_epoch,
			"x-": sum_energy_negative / n_trains_per_epoch,
			"KLD": int(sum_kld / n_trains_per_epoch)
		})
		ddgm.save(args.model_dir)

if __name__ == '__main__':
	main()
