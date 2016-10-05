# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from ddgm import DDGM, activations, DeepGenerativeModel, DeepEnergyModel

class Params():
	def __init__(self, dict=None):
		self.x_width = 40
		self.x_height = self.x_width
		self.x_channels = 3
		self.ndim_z = 10
		self.apply_dropout = False
		self.distribution_x = "universal"	# universal or sigmoid or tanh

		self.energy_model_num_experts = 128
		self.energy_model_feature_extractor_hidden_channels = [64, 128, 256, 512]
		self.energy_model_feature_extractor_stride = 2
		self.energy_model_feature_extractor_ksize = 4
		self.energy_model_feature_extractor_ndim_output = 128
		self.energy_model_batchnorm_to_input = True
		self.energy_model_batchnorm_before_activation = False
		self.energy_model_batchnorm_enabled = True
		self.energy_model_wscale = 1
		self.energy_model_activation_function = "elu"

		self.generative_model_hidden_channels = [512, 256, 128, 64]
		self.generative_model_stride = 2
		self.generative_model_ksize = 4
		self.generative_model_batchnorm_to_input = False
		self.generative_model_batchnorm_before_activation = False
		self.generative_model_batchnorm_enabled = True
		self.generative_model_wscale = 1
		self.generative_model_activation_function = "elu"

		self.gradient_clipping = 10
		self.weight_decay = 0
		self.learning_rate = 0.0001
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
		if self.x_width != self.x_height:
			raise Exception("x_width != x_height")

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

class DCDGM(DDGM):

	def __init__(self, params):
		params.check()
		self.params = params
		self.create_network()
		self.setup_optimizers()

	def create_network(self):
		params = self.params

		# deep energy model
		attributes = {}
		# conv layers
		channels = [(params.x_channels, params.energy_model_feature_extractor_hidden_channels[0])]
		channels += zip(params.energy_model_feature_extractor_hidden_channels[:-1], params.energy_model_feature_extractor_hidden_channels[1:])
		kernel_width = params.energy_model_feature_extractor_ksize
		stride = params.energy_model_feature_extractor_stride
		input_width = params.x_width
		for i, (n_in, n_out) in enumerate(channels):
			pad = input_width % kernel_width % 2
			attributes["layer_%i" % i] = L.Convolution2D(n_in, n_out, kernel_width, stride=stride, pad=pad, wscale=params.energy_model_wscale)
			if params.energy_model_batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)
			input_width = (input_width + pad * 2 - kernel_width) // (kernel_width - stride) + 1

		attributes["b"] = L.Linear(params.x_width * params.x_height * params.x_channels, 1, wscale=params.energy_model_wscale, nobias=True)

		# feature extractor
		n_units_before_projection = input_width * input_width * params.x_channels
		n_units_after_projection = params.energy_model_feature_extractor_ndim_output
		attributes["feature_projector"] = L.Linear(n_units_before_projection, n_units_after_projection, wscale=params.energy_model_wscale)
		if params.energy_model_batchnorm_before_activation:
			attributes["batchnorm_projector"] = L.BatchNormalization(n_units_after_projection)
		else:
			attributes["batchnorm_projector"] = L.BatchNormalization(n_units_before_projection)
		attributes["feature_detector"] = L.Linear(n_units_after_projection, params.energy_model_num_experts, wscale=params.energy_model_wscale)

		self.energy_model = DeepEnergyModel(params, n_layers=len(channels), **attributes)

		# deep generative model
		attributes = {}
		# conv layers
		channels = zip(params.generative_model_hidden_channels[:-1], params.generative_model_hidden_channels[1:])
		channels += [(params.generative_model_hidden_channels[-1], params.x_channels)]
		kernel_width = params.generative_model_ksize
		stride = params.generative_model_stride
		paddings = []
		# compute projection width
		projection_width = params.x_width
		for i, (n_in, n_out) in enumerate(channels):
			pad = projection_width % kernel_width % 2
			paddings = [pad] + paddings
			projection_width = (projection_width + pad * 2 - kernel_width) // (kernel_width - stride) + 1

		for i, (n_in, n_out) in enumerate(channels):
			pad = paddings[i]
			attributes["layer_%i" % i] = L.Deconvolution2D(n_in, n_out, kernel_width, stride=stride, pad=pad, wscale=params.generative_model_wscale)
			if params.generative_model_batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		# projecton layer
		n_units_before_projection = params.ndim_z
		n_units_after_projection = projection_width * projection_width * params.generative_model_hidden_channels[-1]
		attributes["noize_projector"] = L.Linear(n_units_before_projection, n_units_after_projection, wscale=params.generative_model_wscale)
		if params.generative_model_batchnorm_before_activation:
			attributes["batchnorm_projector"] = L.BatchNormalization(n_units_after_projection)
		else:
			attributes["batchnorm_projector"] = L.BatchNormalization(n_units_before_projection)

		if params.distribution_x == "sigmoid":
			self.generative_model = SigmoidDeepGenerativeModel(params, n_layers=len(channels), **attributes)
		elif params.distribution_x == "tanh":
			self.generative_model = TanhDeepGenerativeModel(params, n_layers=len(channels), **attributes)
		elif params.distribution_x == "universal":
			self.generative_model = DeepGenerativeModel(params, n_layers=len(channels), **attributes)
		else:
			raise Exception()

	def setup_optimizers(self):
		params = self.params
		
		opt = optimizers.AdaGrad(lr=params.learning_rate)
		opt.setup(self.energy_model)
		if params.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer_energy_model = opt
		
		opt = optimizers.AdaGrad(lr=params.learning_rate)
		opt.setup(self.generative_model)
		if params.weight_decay > 0:
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer_generative_model = opt

class DeepConvolutionalGenerativeModel(DeepGenerativeModel):

	def project_z(self, z):
		f = activations[self.activation_function]
		if self.batchnorm_enabled == False:
			return f(self.noize_projector(z))
		if self.batchnorm_before_activation:
			return f(self.batchnorm_projector(self.noize_projector(z), test=self.test))
		return self.noize_projector(self.batchnorm_projector(z, test=self.test))

	def __call__(self, z, test=False):
		self.test = test
		return self.compute_output(self.project_z(z))

class SigmoidDeepGenerativeModel(DeepGenerativeModel):
	def __call__(self, x, test=False):
		return F.sigmoid(self.compute_output(self.project_z(z)))

class TanhDeepGenerativeModel(DeepGenerativeModel):
	def __call__(self, x, test=False):
		self.test = test
		return F.tanh(self.compute_output(self.project_z(z)))

class DeepEnergyModel(chainer.Chain):

	def extract_features(self, x):
		f = activations[self.activation_function]
		chain = [x]

		for i in range(self.n_layers):
			u = chain[-1]

			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)

			if self.batchnorm_enabled:
				bn = getattr(self, "batchnorm_%d" % i)
				if i == 0:
					if self.batchnorm_to_input == True:
						u = bn(u, test=self.test)
				else:
					u = bn(u, test=self.test)

			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)

			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not self.test)

			chain.append(output)

		return chain[-1]

	def project_features(self, features):
		if self.batchnorm_enabled == False:
			return F.tanh(self.feature_projector(features))
		if self.batchnorm_before_activation:
			return F.tanh(self.batchnorm_projector(self.feature_projector(features), test=self.test))
		return F.tanh(self.feature_projector(self.batchnorm_projector(features, test=self.test)))

	def compute_energy(self, x, features):
		feature_detector = self.feature_detector(features)

		# avoid overflow
		# -log(1 + exp(x)) = -max(0, x) - log(1 + exp(-|x|)) = -softplus
		experts = -F.softplus(feature_detector)

		sigma = 1.0
		energy = F.sum(x * x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(experts, axis=1)
		
		return energy, experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.extract_features(x)
		energy, experts = self.compute_energy(x, features)
		return energy, experts