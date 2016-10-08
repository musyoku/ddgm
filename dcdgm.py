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
		self.energy_model_feature_extractor_ndim_output = 128
		self.energy_model_feature_extractor_hidden_channels = [64, 128, 256, 512]
		self.energy_model_feature_extractor_stride = 2
		self.energy_model_feature_extractor_ksize = 4
		self.energy_model_batchnorm_to_input = False
		self.energy_model_batchnorm_before_activation = False
		self.energy_model_batchnorm_enabled = False
		self.energy_model_wscale = 1
		self.energy_model_activation_function = "elu"
		self.energy_model_optimizer = "Adam"
		self.energy_model_learning_rate = 0.001
		self.energy_model_momentum = 0.9
		self.energy_model_gradient_clipping = 10
		self.energy_model_weight_decay = 0

		self.generative_model_hidden_channels = [512, 256, 128, 64]
		self.generative_model_stride = 2
		self.generative_model_ksize = 4
		self.generative_model_batchnorm_to_input = False
		self.generative_model_batchnorm_before_activation = True
		self.generative_model_batchnorm_enabled = True
		self.generative_model_wscale = 1
		self.generative_model_activation_function = "elu"
		self.generative_model_optimizer = "Adam"
		self.generative_model_learning_rate = 0.001
		self.generative_model_momentum = 0.9
		self.generative_model_gradient_clipping = 10
		self.generative_model_weight_decay = 0

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

def get_conv_padding(in_size, ksize, stride):
	pad2 = stride - (in_size - ksize) % stride
	if pad2 % stride == 0:
		return 0
	if pad2 % 2 == 1:
		return pad2
	return pad2 / 2

def get_deconv_padding(in_size, out_size, ksize, stride):
	return (stride * (in_size - 1) + ksize - out_size) // 2

def get_deconv_outsize(in_size, ksize, stride, padding):
	return stride * (in_size - 1) + ksize - 2 * padding

def get_deconv_insize(out_size, ksize, stride, padding):
	return (out_size - ksize + 2 * padding) // stride + 1

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
		self.visualize_network()

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
			pad = get_conv_padding(input_width, kernel_width, stride)
			attributes["layer_%i" % i] = L.Convolution2D(n_in, n_out, kernel_width, stride=stride, pad=pad, wscale=params.energy_model_wscale)
			attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out if params.energy_model_batchnorm_before_activation else n_in)
			output_width = (input_width + pad * 2 - kernel_width) // stride + 1
			# check
			if i != len(channels) - 1 and output_width < kernel_width:
				raise Exception("output_width < kernel_width")
			input_width = output_width

		attributes["b"] = L.Linear(params.x_width * params.x_height * params.x_channels, 1, wscale=params.energy_model_wscale, nobias=True)

		# feature extractor
		n_units_before_projection = input_width * input_width * params.energy_model_feature_extractor_hidden_channels[-1]
		n_units_after_projection = params.energy_model_feature_extractor_ndim_output
		attributes["feature_projector"] = L.Linear(n_units_before_projection, n_units_after_projection, wscale=params.energy_model_wscale)
		attributes["batchnorm_projector"] = L.BatchNormalization(n_units_after_projection if params.energy_model_batchnorm_before_activation else n_units_before_projection)
		attributes["feature_detector"] = L.Linear(n_units_after_projection, params.energy_model_num_experts, wscale=params.energy_model_wscale)

		self.energy_model = DeepConvolutionalEnergyModel(params, n_layers=len(channels), **attributes)

		# deep generative model
		attributes = {}
		# conv layers
		channels = zip(params.generative_model_hidden_channels[:-1], params.generative_model_hidden_channels[1:])
		channels += [(params.generative_model_hidden_channels[-1], params.x_channels)]
		kernel_width = params.generative_model_ksize
		stride = params.generative_model_stride
		# compute projection size
		output_width = params.x_width
		for i, (n_in, n_out) in enumerate(channels):
			input_width = get_deconv_insize(output_width, kernel_width, stride, 1)
			# check
			_output_width = get_deconv_outsize(input_width, kernel_width, stride, 1)
			if output_width != _output_width:
				raise Exception("deconvolution missmatch")
			output_width = input_width
		projection_width = input_width

		for i, (n_in, n_out) in enumerate(channels):
			attributes["layer_%i" % i] = L.Deconvolution2D(n_in, n_out, kernel_width, stride=stride, pad=1, wscale=params.generative_model_wscale)
			attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out if params.generative_model_batchnorm_before_activation else n_in)

		# projecton layer
		n_units_before_projection = params.ndim_z
		n_units_after_projection = projection_width * projection_width * params.generative_model_hidden_channels[0]
		attributes["noize_projector"] = L.Linear(n_units_before_projection, n_units_after_projection, wscale=params.generative_model_wscale)
		attributes["batchnorm_projector"] = L.BatchNormalization(n_units_after_projection if params.generative_model_batchnorm_before_activation else n_units_before_projection)

		if params.distribution_x == "sigmoid":
			self.generative_model = SigmoidDeepConvolutionalGenerativeModel(params, n_layers=len(channels), **attributes)
		elif params.distribution_x == "tanh":
			self.generative_model = TanhDeepConvolutionalGenerativeModel(params, n_layers=len(channels), **attributes)
		elif params.distribution_x == "universal":
			self.generative_model = DeepConvolutionalGenerativeModel(params, n_layers=len(channels), **attributes)
		else:
			raise Exception()

		self.generative_model.projection_width = projection_width


	def visualize_network(self):
		params = self.params

		# deep energy model
		print "[Deep Energy Model]"
		print "Input"
		print "| size: {}x{}".format(params.x_width, params.x_height)
		print "v"
		print "Conv Layer:"
		# conv layers
		channels = [(params.x_channels, params.energy_model_feature_extractor_hidden_channels[0])]
		channels += zip(params.energy_model_feature_extractor_hidden_channels[:-1], params.energy_model_feature_extractor_hidden_channels[1:])
		kernel_width = params.energy_model_feature_extractor_ksize
		stride = params.energy_model_feature_extractor_stride
		input_width = params.x_width
		deconv_input_width_array = [input_width]
		for i, (n_in, n_out) in enumerate(channels):
			pad = get_conv_padding(input_width, kernel_width, stride)
			output_width = (input_width + pad * 2 - kernel_width) // stride + 1
			if i != len(channels) - 1 and output_width < kernel_width:
				raise Exception("output_width < kernel_width")
			print "| (ch_in: {}, ch_out: {}, input_size: {}x{} output_size: {}x{} padding: {} stride: {})".format(n_in, n_out, input_width, input_width, output_width, output_width, pad, stride)
			input_width = output_width
			deconv_input_width_array = [input_width] + deconv_input_width_array
		print "v"

		# feature extractor
		n_units_before_projection = input_width * input_width * params.energy_model_feature_extractor_hidden_channels[-1]
		n_units_after_projection = params.energy_model_feature_extractor_ndim_output

		print "Feature Extractor:"
		print "| {}x{}x{} -> [projection] -> {}".format(input_width, input_width, params.energy_model_feature_extractor_hidden_channels[-1], n_units_after_projection)
		print "v"
		print "Product of Experts:"
		print "| {} -> {}".format(n_units_after_projection, params.energy_model_num_experts)
		print "v"
		print "Energy\n"

		# deep generative model
		print "[Deep Generative Model]"
		print "z"
		print "| ndim: {}".format(params.ndim_z)
		print "v"

		# conv layers
		channels = zip(params.generative_model_hidden_channels[:-1], params.generative_model_hidden_channels[1:])
		channels += [(params.generative_model_hidden_channels[-1], params.x_channels)]
		kernel_width = params.generative_model_ksize
		stride = params.generative_model_stride
		# compute projection size
		output_width = params.x_width
		deconv_input_width_array = [output_width]
		for i, (n_in, n_out) in enumerate(channels):
			input_width = get_deconv_insize(output_width, kernel_width, stride, 1)
			output_width = input_width
			deconv_input_width_array = [output_width] + deconv_input_width_array
		projection_width = input_width

		print "Projection Layer:"
		print "| ndim: {} -> [projection] -> size: {}x{}, ch: {}".format(params.ndim_z, projection_width, projection_width, params.generative_model_hidden_channels[0])
		print "v"
		print "Deconv Layer:"

		for i, (n_in, n_out) in enumerate(channels):
			input_width = deconv_input_width_array[i]
			output_width = deconv_input_width_array[i + 1]
			print "| (ch_in: {}, ch_out: {}, input_size: {}x{} output_size: {}x{} padding: {}, stride: {})".format(n_in, n_out, input_width, input_width, output_width, output_width, 1, stride)

		print "v"
		print "x~"

class DeepConvolutionalGenerativeModel(DeepGenerativeModel):

	def project_z(self, z):
		f = activations[self.activation_function]
		if self.batchnorm_enabled == False:
			projection = self.noize_projector(z)
		else:
			if self.batchnorm_before_activation:
				projection = self.batchnorm_projector(self.noize_projector(z), test=self.test)
			else:
				projection = self.noize_projector(self.batchnorm_projector(z, test=self.test))
		projection = f(projection)
		batchsize = projection.data.shape[0]
		return F.reshape(projection, (batchsize, -1, self.projection_width, self.projection_width))

	def __call__(self, z, test=False):
		self.test = test
		return self.compute_output(self.project_z(z))

class SigmoidDeepConvolutionalGenerativeModel(DeepConvolutionalGenerativeModel):
	def __call__(self, z, test=False):
		return F.sigmoid(self.compute_output(self.project_z(z)))

class TanhDeepConvolutionalGenerativeModel(DeepConvolutionalGenerativeModel):
	def __call__(self, z, test=False):
		self.test = test
		return F.tanh(self.compute_output(self.project_z(z)))

class DeepConvolutionalEnergyModel(DeepEnergyModel):

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
		batchsize = x.data.shape[0]
		_x = F.reshape(x, (batchsize, -1))
		energy = F.sum(_x * _x, axis=1) / sigma - F.reshape(self.b(x), (-1,)) + F.sum(experts, axis=1)
		
		return energy, experts

	def __call__(self, x, test=False):
		self.test = test
		features = self.extract_features(x)
		batchsize = features.data.shape[0]
		features = F.reshape(features, (batchsize, -1))
		features = self.project_features(features)
		energy, experts = self.compute_energy(x, features)
		return energy, experts