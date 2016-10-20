# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from softplus import softplus
from params import Params
import weightnorm as WN

nonlinearities = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

class EnergyModelParams(Params):
	def __init__(self, dict=None):
		self.ndim_input = 28 * 28
		self.ndim_output = 10

		self.num_experts = 128
		self.feature_extractor_hidden_units = [800, 800]
		self.use_batchnorm = True
		self.batchnorm_to_input = True

		# True:  y = f(BN(W*x + b))
		# False: y = f(W*BN(x) + b))
		self.batchnorm_before_activation = False

		self.use_dropout_at_layer = [False, False]	# Do not apply dropout to output layer
		self.add_output_noise_at_layer = [True, True]
		self.use_weightnorm = True
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

class GenerativeModelParams(Params):
	def __init__(self, dict=None):
		self.ndim_input = 10
		self.ndim_output = 28 * 28
		self.distribution_output = "universal"	# universal or sigmoid or tanh

		self.hidden_units = [800, 800]
		self.use_dropout_at_layer = [False, False]	# Do not apply dropout to output layer
		self.add_output_noise_at_layer = [False, False]
		self.use_batchnorm = True
		self.batchnorm_to_input = False
		self.batchnorm_before_activation = False
		self.use_weightnorm = False
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class DDGM():

	def __init__(self, params_energy_model, params_generative_model):
		params_energy_model.check()
		params_generative_model.check()
		self.params = Params(name="DDGM")
		self.params.energy_model = params_energy_model
		self.params.generative_model = params_generative_model
		self.create_network()
		self.setup_optimizers()

	def get_initializer(self, name, std):
		if name.lower() == "normal":
			initialW = initializers.Normal(std)
		elif name.lower() == "glorotnormal":
			initialW = initializers.GlorotNormal(std)
		elif name.lower() == "henormal":
			initialW = initializers.HeNormal(std)
		else:
			raise Exception()

	def get_optimizer(self, name, lr, momentum):
		if name.lower() == "adam":
			return optimizers.Adam(alpha=lr, beta1=momentum)
		if name.lower() == "adagrad":
			return optimizers.AdaGrad(lr=lr)
		if name.lower() == "adadelta":
			return optimizers.AdaDelta(rho=momentum)
		if name.lower() == "nesterov" or name.lower() == "nesterovag":
			return optimizers.NesterovAG(lr=lr, momentum=momentum)
		if name.lower() == "rmsprop":
			return optimizers.RMSprop(lr=lr, alpha=momentum)
		if name.lower() == "momentumsgd":
			return optimizers.MomentumSGD(lr=lr, mommentum=mommentum)
		if name.lower() == "sgd":
			return optimizers.SGD(lr=lr)
			
	def get_linear_layer(self, use_weightnorm, initializer, init_std):
		if use_weightnorm:
			return WN.Linear(n_in, n_out, initialV=self.get_initializer(initializer, init_std))
		return L.Linear(n_in, n_out, initialW=self.get_initializer(initializer, init_std))

	def create_network(self):
		self.create_energy_model()
		self.create_generative_model()

	def create_energy_model(self):
		params = self.params.energy_model

		attributes = {}
		units = [(params.ndim_input, params.feature_extractor_hidden_units[0])]
		units += zip(params.feature_extractor_hidden_units[:-1], params.feature_extractor_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = self.get_linear_layer(params.use_weightnorm, params.weight_initializer, params.weight_init_std)
			if params.use_batchnorm:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out if params.batchnorm_before_activation else n_in)

		attributes["b"] = L.Linear(params.ndim_input, 1, initialW=self.get_initializer(params.weight_initializer, params.weight_init_std), nobias=True)
		attributes["feature_detector"] = L.Linear(params.feature_extractor_hidden_units[-1], params.num_experts, initialW=self.get_initializer(params.weight_initializer, params.weight_init_std))
		
		params.n_layers = len(units)
		self.energy_model = DeepEnergyModel(params, **attributes)

	def create_generative_model(self):
		params = self.params.generative_model

		attributes = {}
		units = [(params.ndim_input, params.hidden_units[0])]
		units += zip(params.hidden_units[:-1], params.hidden_units[1:])
		units += [(params.hidden_units[-1], params.ndim_output)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = self.get_linear_layer(params.use_weightnorm, params.weight_initializer, params.weight_init_std)
			if params.use_batchnorm:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out if params.batchnorm_before_activation else n_in)

		params.n_layers = len(units)
		if params.distribution_output == "sigmoid":
			self.generative_model = SigmoidDeepGenerativeModel(params, **attributes)
		elif params.distribution_output == "tanh":
			self.generative_model = TanhDeepGenerativeModel(params, **attributes)
		elif params.distribution_output == "universal":
			self.generative_model = DeepGenerativeModel(params, **attributes)
		else:
			raise Exception()

	def setup_optimizers(self):
		params = self.params
		
		opt = self.get_optimizer(params.optimizer, params.learning_rate, params.momentum)
		opt.setup(self.energy_model)
		if params.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer_energy_model = opt
		
		opt = self.get_optimizer(params.optimizer, params.learning_rate, params.momentum)
		opt.setup(self.generative_model)
		if params.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer_generative_model = opt

	def update_laerning_rate(self, lr):
		self.optimizer_energy_model.alpha = lr
		self.optimizer_generative_model.alpha = lr

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self.params.gpu_enabled

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x.to_cpu()
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		if isinstance(x, Variable):
			return x.data.shape[0]
		return x.shape[0]

	def zero_grads(self):
		self.optimizer_energy_model.zero_grads()
		self.optimizer_generative_model.zero_grads()

	def compute_energy(self, x_batch, test=False):
		x_batch = self.to_variable(x_batch)
		return self.energy_model(x_batch, test=test)

	def compute_entropy(self):
		return self.generative_model.compute_entropy()

	def sample_z(self, batchsize=1):
		# uniform
		z_batch = np.random.uniform(-1, 1, (batchsize, self.params.ndim_z)).astype(np.float32)
		# gaussian
		# z_batch = np.random.normal(0, 1, (batchsize, self.params.ndim_z)).astype(np.float32)
		return z_batch

	def generate_x(self, batchsize=1, test=False, as_numpy=False):
		return self.generate_x_from_z(self.sample_z(batchsize), test=test, as_numpy=as_numpy)

	def generate_x_from_z(self, z_batch, test=False, as_numpy=False):
		z_batch = self.to_variable(z_batch)
		x_batch = self.generative_model(z_batch, test=test)
		if as_numpy:
			return self.to_numpy(x_batch)
		return x_batch

	def backprop_energy_model(self, loss):
		self.zero_grads()
		loss.backward()
		self.optimizer_energy_model.update()

	def backprop_generative_model(self, loss):
		self.zero_grads()
		loss.backward()
		self.optimizer_generative_model.update()

	def compute_kld_between_generator_and_energy_model(self, x_batch_negative):
		energy_negative, experts_negative = self.compute_energy(x_batch_negative)
		entropy = self.generative_model.compute_entropy()
		return F.sum(energy_negative) / self.get_batchsize(x_batch_negative) - entropy

	def compute_loss(self, x_batch_positive, x_batch_negative):
		energy_positive, experts_positive = self.compute_energy(x_batch_positive)
		energy_negative, experts_negative = self.compute_energy(x_batch_negative)
		energy_positive = F.sum(energy_positive) / self.get_batchsize(x_batch_positive)
		energy_negative = F.sum(energy_negative) / self.get_batchsize(x_batch_negative)
		loss = energy_positive - energy_negative
		return loss, energy_positive, energy_negative

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/{}.hdf5".format(attr)
				if os.path.isfile(filename):
					print "loading",  filename
					serializers.load_hdf5(filename, prop)
				else:
					print filename, "not found."

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		for attr in vars(self):
			prop = getattr(self, attr)
			if isinstance(prop, chainer.Chain) or isinstance(prop, chainer.optimizer.GradientMethod):
				filename = dir + "/{}.hdf5".format(attr)
				if os.path.isfile(filename):
					os.remove(filename)
				serializers.save_hdf5(filename, prop)
		print "model saved."

class DeepGenerativeModel(chainer.Chain):
	def __init__(self, params, **layers):
		super(DeepGenerativeModel, self).__init__(**layers)

		self.n_layers = params.n_layers
		self.nonlinearity = params.nonlinearity
		self.use_dropout_at_layer = params.use_dropout_at_layer
		self.add_noise_at_layer = params.add_output_noise_at_layer
		self.use_batchnorm = params.use_batchnorm
		self.batchnorm_to_input = params.batchnorm_to_input
		self.batchnorm_before_activation = params.batchnorm_before_activation

		if params.gpu_enabled:
			self.to_gpu()

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def compute_entropy(self):
		entropy = 0

		if self.use_batchnorm == False:
			return entropy

		for i in range(self.n_layers):
			bn = getattr(self, "batchnorm_%d" % i)
			entropy += F.sum(F.log(2 * math.e * math.pi * bn.gamma ** 2 + 1e-8) / 2)

		return entropy

	def compute_output(self, z):
		f = nonlinearities[self.nonlinearity]
		chain = [z]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.use_batchnorm:
				bn = getattr(self, "batchnorm_%d" % i)
				if i == 0:
					if self.batchnorm_to_input == True:
						u = bn(u, test=self.test)
				elif i == self.n_layers - 1:
					if self.batchnorm_before_activation == False:
						u = bn(u, test=self.test)
				else:
					u = bn(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.use_dropout_at_layer[i]:
					output = F.dropout(output, train=not self.test)

			chain.append(output)

		return chain[-1]

	def __call__(self, z, test=False):
		self.test = test
		return self.compute_output(z)

class SigmoidDeepGenerativeModel(DeepGenerativeModel):
	def __call__(self, z, test=False):
		self.test = test
		return F.sigmoid(self.compute_output(z))

class TanhDeepGenerativeModel(DeepGenerativeModel):
	def __call__(self, z, test=False):
		self.test = test
		return F.tanh(self.compute_output(z))

class DeepEnergyModel(chainer.Chain):
	def __init__(self, params, **layers):
		super(DeepEnergyModel, self).__init__(**layers)

		self.n_layers = params.n_layers
		self.nonlinearity = params.nonlinearity
		self.use_dropout_at_layer = params.use_dropout_at_layer
		self.add_noise_at_layer = params.add_output_noise_at_layer
		self.use_batchnorm = params.use_batchnorm
		self.batchnorm_to_input = params.batchnorm_to_input
		self.batchnorm_before_activation = params.batchnorm_before_activation

		if params.gpu_enabled:
			self.to_gpu()

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def extract_features(self, x):
		f = nonlinearities[self.nonlinearity]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.use_batchnorm:
				bn = getattr(self, "batchnorm_%d" % i)
				if i == 0:
					if self.batchnorm_to_input == True:
						u = bn(u, test=self.test)
				else:
					u = bn(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			if i == self.n_layers - 1:
				output = F.tanh(u)
			else:
				output = f(u)
				if self.use_dropout_at_layer[i]:
					output = F.dropout(output, train=not self.test)

			chain.append(output)

		return chain[-1]

	def compute_energy(self, x, features):
		feature_detector = self.feature_detector(features)

		# avoid overflow
		# -log(1 + exp(x)) = -max(0, x) - log(1 + exp(-|x|)) = -softplus
		experts = -softplus(feature_detector)

		sigma = 1.0
		energy = F.sum(x * x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(experts, axis=1)
		
		return energy, experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.extract_features(x)
		energy, experts = self.compute_energy(x, features)
		return energy, experts