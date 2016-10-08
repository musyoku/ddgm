from scipy.io import wavfile
import numpy as np
import os, sys, time, random, math, pylab
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from model import params, ddgm
from args import args
import sampler

def plot(z, color="blue", s=40):
	for n in xrange(z.shape[0]):
		result = pylab.scatter(z[n, 0], z[n, 1], s=s, marker="o", edgecolors="none", color=color)
	ax = pylab.subplot(111)
	ax.set_xlim(-4, 4)
	ax.set_ylim(-4, 4)
	pylab.xticks(pylab.arange(-4, 5))
	pylab.yticks(pylab.arange(-4, 5))


def main():
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	# settings
	max_epoch = 1000
	n_trains_per_epoch = 50
	batchsize_positive = 100
	batchsize_negative = 100
	plotsize = 400

	# seed
	np.random.seed(args.seed)
	if params.gpu_enabled:
		cuda.cupy.random.seed(args.seed)

	fixed_z = ddgm.sample_z(plotsize)
	fixed_target = sampler.sample_from_swiss_roll(600, params.ndim_x, 10)

	total_time = 0
	for epoch in xrange(1, max_epoch):
		sum_energy_positive = 0
		sum_energy_negative = 0
		sum_kld = 0
		epoch_time = time.time()

		for t in xrange(n_trains_per_epoch):
			# sample from data distribution
			x_positive = sampler.sample_from_swiss_roll(batchsize_positive, params.ndim_x, 10)

			# train energy model
			x_negative = ddgm.generate_x(batchsize_negative)
			loss, energy_positive, energy_negative = ddgm.compute_loss(x_positive, x_negative)
			ddgm.backprop_energy_model(loss)

			# train generative model
			x_negative = ddgm.generate_x(batchsize_negative)
			kld = ddgm.compute_kld_between_generator_and_energy_model(x_negative)
			ddgm.backprop_generative_model(kld)

			sum_energy_positive += float(energy_positive.data)
			sum_energy_negative += float(energy_negative.data)
			sum_kld += float(kld.data)
			if t % 10 == 0:
				sys.stdout.write("\rTraining in progress...({} / {})".format(t, n_trains_per_epoch))
				sys.stdout.flush()

		epoch_time = time.time() - epoch_time
		total_time += epoch_time
		sys.stdout.write("\r")
		print "epoch: {} loss: x+ {:.3f} x- {:.3f} kld: {:.3f} time: {} min total: {} min".format(epoch, sum_energy_positive / n_trains_per_epoch, sum_energy_negative / n_trains_per_epoch, sum_kld / n_trains_per_epoch, int(epoch_time / 60), int(total_time / 60))
		sys.stdout.flush()
		ddgm.save(args.model_dir)

		# init
		fig = pylab.gcf()
		fig.set_size_inches(8.0, 8.0)
		pylab.clf()

		plot(fixed_target, color="#bec3c7", s=20)
		plot(ddgm.generate_x_from_z(fixed_z, as_numpy=True, test=True), color="#e84c3d", s=20)

		# save
		pylab.savefig("{}/{}.png".format(args.plot_dir, 100000 + epoch))
		

if __name__ == '__main__':
	main()
