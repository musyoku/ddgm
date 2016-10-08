import numpy as np
import os, sys, time, random, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from model import params, ddgm
from args import args
import sampler

def main():
	# settings
	max_epoch = 1000
	n_trains_per_epoch = 500
	batchsize_positive = 100
	batchsize_negative = 100

	# seed
	np.random.seed(args.seed)
	if params.gpu_enabled:
	    cuda.cupy.random.seed(args.seed)

	total_time = 0
	for epoch in xrange(1, max_epoch):
		sum_energy_positive = 0
		sum_energy_negative = 0
		sum_kld = 0
		epoch_time = time.time()

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sampler.sample_from_gaussian_mixture(batchsize_positive, params.ndim_x, 10)

			# train energy model
			x_negative = ddgm.generate_x(batchsize_negative)
			loss, positive_energy, negative_energy = ddgm.compute_loss(x_positive, x_negative)
			# loss := positive_energy - negative_energy
			ddgm.backprop_energy_model(positive_energy)
			ddgm.backprop_energy_model(-negative_energy)
			# ddgm.backprop_energy_model(loss)

			# train generative model
			x_negative = ddgm.generate_x(batchsize_negative)
			kld = ddgm.compute_kld_between_generator_and_energy_model(x_negative)
			ddgm.backprop_generative_model(kld)

			sum_energy_positive += float(positive_energy.data)
			sum_energy_negative += float(negative_energy.data)
			sum_kld += float(kld.data)
			if t % 10 == 0:
				sys.stdout.write("\rTraining in progress...({} / {})".format(t, n_trains_per_epoch))
				sys.stdout.flush()

		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch: {} loss: {:.3f} {:.3f} kld: {:.3f} time: {} min total: {} min".format(epoch, sum_energy_positive / n_trains_per_epoch, sum_energy_negative / n_trains_per_epoch, sum_kld / n_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
		sys.stdout.flush()
		ddgm.save(args.model_dir)

if __name__ == '__main__':
	main()
