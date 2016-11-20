# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time, copy
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from softplus import softplus
from params import Params
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class EnergyModelParams(Params):
	def __init__(self):
		self.ndim_input = 28 * 28
		self.ndim_output = 10
		self.num_experts = 128
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

class GenerativeModelParams(Params):
	def __init__(self):
		self.ndim_input = 10
		self.ndim_output = 28 * 28
		self.distribution_output = "universal"	# universal or sigmoid or tanh
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

class DDGM():

	def __init__(self, params_energy_model, params_generative_model):
		self.params_energy_model = copy.deepcopy(params_energy_model)
		self.config_energy_model = to_object(params_energy_model["config"])
		self.params_generative_model = copy.deepcopy(params_generative_model)
		self.config_generative_model = to_object(params_generative_model["config"])
		self.build_network()
		self._gpu = False

	def build_network(self):
		self.build_energy_model()
		self.build_generative_model()

	def build_energy_model(self):
		params = self.params_energy_model
		self.energy_model = DeepEnergyModel()
		self.energy_model.add_feature_extractor(sequential.from_dict(params["feature_extractor"]))
		self.energy_model.add_experts(sequential.from_dict(params["experts"]))
		self.energy_model.add_b(sequential.from_dict(params["b"]))
		config = self.config_energy_model
		self.energy_model.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)

	def build_generative_model(self):
		params = self.params_generative_model
		self.generative_model = DeepGenerativeModel()
		self.generative_model.add_sequence(sequential.from_dict(params["model"]))
		config = self.config_generative_model
		self.generative_model.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)

	def to_gpu(self):
		self.energy_model.to_gpu()
		self.generative_model.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

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

	# returns energy and product of experts
	def compute_energy(self, x_batch, test=False):
		x_batch = self.to_variable(x_batch)
		return self.energy_model(x_batch, test=test)

	def compute_energy_sum(self, x_batch, test=False):
		energy, experts = self.compute_energy(x_batch, test)
		energy = F.sum(energy) / self.get_batchsize(x_batch)
		return energy

	def compute_entropy(self):
		return self.generative_model.compute_entropy()

	def sample_z(self, batchsize=1):
		config = self.config_generative_model
		ndim_z = config.ndim_input
		# uniform
		z_batch = np.random.uniform(-1, 1, (batchsize, ndim_z)).astype(np.float32)
		# gaussian
		# z_batch = np.random.normal(0, 1, (batchsize, ndim_z)).astype(np.float32)
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
		self.energy_model.backprop(loss)

	def backprop_generative_model(self, loss):
		self.generative_model.backprop(loss)

	def compute_kld_between_generator_and_energy_model(self, x_batch_negative):
		energy_negative, experts_negative = self.compute_energy(x_batch_negative)
		entropy = self.generative_model.compute_entropy()
		return F.sum(energy_negative) / self.get_batchsize(x_batch_negative) - entropy

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.energy_model.load(dir + "/energy_model.hdf5")
		self.generative_model.load(dir + "/generative_model.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.energy_model.save(dir + "/energy_model.hdf5")
		self.generative_model.save(dir + "/generative_model.hdf5")

class DeepGenerativeModel(sequential.chain.Chain):

	def compute_entropy(self):
		entropy = 0
		for i, link in enumerate(self.sequence.links):
			if isinstance(link, L.BatchNormalization):
				entropy += F.sum(F.log(2 * math.e * math.pi * link.gamma ** 2 + 1e-8) / 2)
		return entropy

	def __call__(self, z, test=False):
		return self.sequence(z, test=test)

class DeepEnergyModel(sequential.chain.Chain):

	def add_feature_extractor(self, sequence):
		self.add_sequence_with_name(sequence, "feature_extractor")
		self.feature_extractor = sequence

	def add_experts(self, sequence):
		self.add_sequence_with_name(sequence, "experts")
		self.experts = sequence

	def add_b(self, sequence):
		self.add_sequence_with_name(sequence, "b")
		self.b = sequence

	def compute_energy(self, x, features):
		experts = self.experts(features)

		# avoid overflow
		# -log(1 + exp(x)) = -max(0, x) - log(1 + exp(-|x|)) = -softplus
		product_of_experts = -softplus(experts)

		sigma = 1.0
		if x.data.ndim == 4:
			batchsize = x.data.shape[0]
			_x = F.reshape(x, (batchsize, -1))
			energy = F.sum(_x * _x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(product_of_experts, axis=1)
		else:
			energy = F.sum(x * x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(product_of_experts, axis=1)
		
		return energy, product_of_experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.feature_extractor(x, test=test)
		energy, product_of_experts = self.compute_energy(x, features)
		return energy, product_of_experts