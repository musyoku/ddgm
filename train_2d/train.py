from scipy.io import wavfile
import numpy as np
import os, sys, time, random, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from model import params, ddgm
from args import args
from dataset import binarize_data, load_images
import dataset

def sample_from_data(batchsize, n_dim, n_labels):
	def sample(label, n_labels):
		uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
		r = math.sqrt(uni) * 3.0
		rad = np.pi * 4.0 * math.sqrt(uni)
		x = r * math.cos(rad)
		y = r * math.sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, n_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(n_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(random.randint(0, n_labels - 1), n_labels)
	
	return z

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
		sum_loss = 0
		sum_kld = 0
		epoch_time = time.time()

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sample_from_data(batchsize_positive, params.ndim_x, 10)

			# train energy model
			## sample from generator
			x_negative = ddgm.generate_x(batchsize_negative)
			## positive phase
			loss_positive = ddgm.compute_positive_loss(x_positive)
			ddgm.backprop_energy_model(loss_positive)
			## negative phase
			loss_negative = ddgm.compute_negative_loss(x_negative)
			ddgm.backprop_energy_model(loss_negative)

			# train generative model
			## sample from generator
			x_negative = ddgm.generate_x(batchsize_negative)
			kld = ddgm.compute_kld_between_generator_and_energy_model(x_negative)
			## update parameters
			ddgm.backprop_generative_model(kld)

			sum_loss += float(loss_positive.data + loss_negative.data)
			sum_kld += float(kld.data)
			if t % 10 == 0:
				sys.stdout.write("\rTraining in progress...({} / {})".format(t, n_trains_per_epoch))
				sys.stdout.flush()

		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch: {} loss: {:.3f} kld: {:.3f} time: {} min total: {} min".format(epoch + 1, sum_loss / n_trains_per_epoch, sum_kld / n_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
		sys.stdout.flush()
		ddgm.save(args.model_dir)


		# validation
		x_positive = sample_from_data(batchsize_positive, params.ndim_x, 10)
		energy, experts = ddgm.compute_energy(x_positive, test=True)
		energy.to_cpu()
		experts.to_cpu()
		print "energy (pos):", np.mean(energy.data) 
		# print "logP(x):", -np.sum(experts.data, axis=1)
		x_negative = ddgm.generate_x(batchsize_negative)
		energy, experts = ddgm.compute_energy(x_negative, test=True)
		energy.to_cpu()
		experts.to_cpu()
		print "energy (neg): ", np.mean(energy.data) 
		# print "logP(x):", -np.sum(experts.data, axis=1)

if __name__ == '__main__':
	main()
