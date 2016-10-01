import sys, os, random, math
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import params, ddgm

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
	try:
		os.mkdir(args.plot_dir)
	except:
		pass

	x_positive = sample_from_data(100, 2, 10)
	visualizer.plot_z(x_positive, dir=args.plot_dir, filename="positive")

	x_negative = ddgm.generate_x(100)
	if params.gpu_enabled:
		x_negative.to_cpu()
	visualizer.plot_z(x_negative.data, dir=args.plot_dir, filename="negative")

if __name__ == '__main__':
	main()
