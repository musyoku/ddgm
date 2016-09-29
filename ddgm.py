# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

class Params():
	def __init__(self, dict=None):
		self.ndim_x = 28 * 28
		self.ndim_z = 10
		self.batchnorm_before_activation = True
		self.batchnorm_enabled = True
		self.activation_function = "elu"
		self.apply_dropout = False

		self.energy_model_num_experts = 128
		self.energy_model_features_hidden_units = [500]
		self.energy_model_apply_batchnorm_to_input = True

		self.generative_model_hidden_units = [500]
		self.generative_model_apply_batchnorm_to_input = False

		self.wscale = 0.1
		self.gradient_clipping = 10
		self.gradient_momentum = 0.9
		self.weight_decay = 0
		self.learning_rate = 0.001
		self.gpu_enabled = True

		if dict:
			self.from_dict(dict)
			self.check()

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

	def dump(self):
		print "params:"
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

	def check(self):
		base = Params()
		for attr, value in self.__dict__.iteritems():
			if not hasattr(base, attr):
				raise Exception("invalid parameter '{}'".format(attr))

class DDGM():

	def __init__(self, params):
		params.check()
		self.params = params
		self.create_network()
		self.setup_optimizers()

	def create_network(self):
		params = self.params

		# deep energy model
		attributes = {}
		units = [(params.ndim_x, params.energy_model_features_hidden_units[0])]
		units += zip(params.energy_model_features_hidden_units[:-1], params.energy_model_features_hidden_units[1:])
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=params.wscale)
			if params.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		attributes["b"] = L.Linear(params.ndim_x, 1, wscale=params.wscale, nobias=True)
		attributes["experts"] = L.Linear(params.energy_model_features_hidden_units[-1], params.energy_model_num_experts, wscale=params.wscale)

		self.energy_model = DeepEnergyModel(params, n_layers=len(units), **attributes)

		# deep generative model
		attributes = {}
		units = [(params.ndim_z, params.generative_model_hidden_units[0])]
		units += zip(params.generative_model_hidden_units[:-1], params.generative_model_hidden_units[1:])
		units += [(params.generative_model_hidden_units[-1], params.ndim_x)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=params.wscale)
			if params.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		self.generative_model = DeepGenerativeModel(params, n_layers=len(units), **attributes)

	def setup_optimizers(self):
		params = self.params
		
		opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
		opt.setup(self.energy_model)
		if params.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer_energy_model = opt
		
		opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
		opt.setup(self.generative_model)
		if params.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer_generative_model = opt

	def update_laerning_rate(self, lr):
		self.optimizer_energy_model.alpha = lr
		self.optimizer_generative_model.alpha = lr

	def zero_grads(self):
		self.optimizer_energy_model.zero_grads()
		self.optimizer_generative_model.zero_grads()

	def update(self):
		self.optimizer_energy_model.update()
		self.optimizer_generative_model.update()

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

	def get_batchsize(self, x):
		if isinstance(x, Variable):
			return x.data.shape[0]
		return x.shape[0]

	def compute_energy(self, x_batch, test=False):
		x_batch = self.to_variable(x_batch)
		return self.energy_model(x_batch, test=test)

	def sample_z(self, batchsize=1):
		# uniform
		z_batch = np.random.uniform(-1, 1, (batchsize, self.params.ndim_z)).astype(np.float32)
		# gaussian
		# z_batch = np.random.normal(0, 1, (batchsize, self.params.ndim_z)).astype(np.float32)
		return z_batch

	def generate_x(self, batchsize=1, test=False):
		return self.generate_x_from_z(self.sample_z(batchsize), test=test)

	def generate_x_from_z(self, z_batch, test=False):
		z_batch = self.to_variable(z_batch)
		return self.generative_model(z_batch)

	def backprop(self, loss):
		self.zero_grads()
		loss.backward()
		self.update()

	def compute_loss(self, x_batch_positive, x_batch_negative):
		energy_positive, experts_positive = self.compute_energy(x_batch_positive)
		energy_negative, experts_negative = self.compute_energy(x_batch_negative)
		return F.sum(energy_positive + energy_negative)

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
		print "model loaded."

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
				serializers.save_hdf5(dir + "/{}.hdf5".format(attr), prop)
		print "model saved."

class DeepGenerativeModel(chainer.Chain):
	def __init__(self, params, n_layers, **layers):
		super(DeepGenerativeModel, self).__init__(**layers)

		self.n_layers = n_layers
		self.activation_function = params.activation_function
		self.apply_dropout = params.apply_dropout
		self.apply_batchnorm = params.batchnorm_enabled
		self.apply_batchnorm_to_input = params.energy_model_apply_batchnorm_to_input
		self.batchnorm_before_activation = params.batchnorm_before_activation

		if params.gpu_enabled:
			self.to_gpu()

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def compute_output(self, x):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input == True:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
				else:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not self.test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False):
		self.test = test
		return self.compute_output(x)

class DeepEnergyModel(chainer.Chain):
	def __init__(self, params, n_layers, **layers):
		super(DeepEnergyModel, self).__init__(**layers)

		self.n_layers = n_layers
		self.activation_function = params.activation_function
		self.apply_dropout = params.apply_dropout
		self.apply_batchnorm = params.batchnorm_enabled
		self.apply_batchnorm_to_input = params.energy_model_apply_batchnorm_to_input
		self.batchnorm_before_activation = params.batchnorm_before_activation

		if params.gpu_enabled:
			self.to_gpu()

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def extract_features(self, x):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input == True:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
				else:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not self.test)
			chain.append(output)

		return chain[-1]

	def compute_energy(self, x, features):
		experts = self.experts(features)
		print "experts"
		print experts.data
		print "exp"
		print F.exp(experts).data
		experts = -F.log(1 + F.exp(experts))
		print "log"
		print experts.data
		value = float(F.sum(experts).data)
		print "energy"
		print value
		if value != value:
			raise Exception()

		sigma = 1.0
		energy = F.reshape(F.sum(x * x, axis=1), (-1, 1)) / sigma - self.b(x) + F.reshape(F.sum(experts, axis=1), (-1, 1))
		return energy, experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.extract_features(x)
		energy, experts = self.compute_energy(x, features)
		return energy, experts