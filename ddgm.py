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
		self.batchnorm_before_activation = True
		self.batchnorm_enabled = True
		self.activation_function = "elu"
		self.apply_dropout = False

		self.energy_model_num_experts = 128
		self.energy_model_features_hidden_units = [128]
		self.energy_model_apply_batchnorm_to_input = True

		self.generative_model_ndim_z = 10
		self.generative_model_hidden_units = [128]
		self.generative_model_apply_batchnorm_to_input = False

		self.wscale = 0.1


		if dict:
			self.from_dict(dict)

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
		pass

class DDGM():

	def __init__(self, params):
		params.check()
		self.params = params
		self.create_network()
		self.setup_optimizers()

	def create_network(self):
		params = self.params

		# deep energy model
		units = [(params.ndim_x, params.energy_model_features_hidden_units[0])]
		units += zip(params.energy_model_features_hidden_units[:-1], params.energy_model_features_hidden_units[1:])
		units += [(params.energy_model_features_hidden_units[-1], params.energy_model_num_experts)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=params.wscale)
			if params.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		energy_model = DeepGenerativeModel(**attributes, params, len(units))

		# deep generative model
		units = [(params.generative_model_ndim_z, params.generative_model_hidden_units[0])]
		units += zip(params.generative_model_hidden_units[:-1], params.generative_model_hidden_units[1:])
		units += [(params.generative_model_hidden_units[-1], params.ndim_x)]
		for i, (n_in, n_out) in enumerate(units):
			attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=params.wscale)
			if params.batchnorm_before_activation:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)
			else:
				attributes["batchnorm_%i" % i] = L.BatchNormalization(n_in)

		generative_model = DeepGenerativeModel(**attributes, params, len(units))


	def setup_optimizers(self):
		params = self.params
		
		self.causal_conv_optimizers = []
		for layer in self.causal_conv_layers:
			opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
			opt.setup(layer)
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
			opt.add_hook(GradientClipping(params.gradient_clipping))
			self.causal_conv_optimizers.append(opt)
		
		self.residual_conv_optimizers = []
		for block in self.residual_blocks:
			for layer in block:
				opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
				opt.setup(layer)
				opt.add_hook(optimizer.WeightDecay(params.weight_decay))
				opt.add_hook(GradientClipping(params.gradient_clipping))
				self.residual_conv_optimizers.append(opt)
		
		self.softmax_conv_optimizers = []
		for layer in self.softmax_conv_layers:
			opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
			opt.setup(layer)
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
			opt.add_hook(GradientClipping(params.gradient_clipping))
			self.softmax_conv_optimizers.append(opt)

	def update_laerning_rate(self, lr):
		for opt in self.causal_conv_optimizers:
			opt.alpha = lr

		for opt in self.residual_conv_optimizers:
			opt.alpha = lr
			
		for opt in self.softmax_conv_optimizers:
			opt.alpha = lr

	def zero_grads(self):
		for opt in self.causal_conv_optimizers:
			opt.zero_grads()

		for opt in self.residual_conv_optimizers:
			opt.zero_grads()
			
		for opt in self.softmax_conv_optimizers:
			opt.zero_grads()

	def update(self):
		for opt in self.causal_conv_optimizers:
			opt.update()

		for opt in self.residual_conv_optimizers:
			opt.update()
			
		for opt in self.softmax_conv_optimizers:
			opt.update()

	@property
	def gpu_enabled(self):
		return self.params.gpu_enabled

	def slice_1d(self, x, cut=0):
		return CausalSlice1d(cut)(x)

	def padding_1d(self, x, pad=0):
		return CausalPadding1d(pad)(x)

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

	def forward_one_step(self, x_batch, softmax=True, return_numpy=False):
		x_batch = self.to_variable(x_batch)
		causal_output = self.forward_causal_block(x_batch)
		residual_output, sum_skip_connections = self.forward_residual_block(causal_output)
		softmax_output = self.forward_softmax_block(sum_skip_connections, softmax=softmax)
		if return_numpy:
			if self.gpu_enabled:
				softmax_output.to_cpu()
			return softmax_output.data
		return softmax_output

	def forward_causal_block(self, x_batch):
		input_batch = self.to_variable(x_batch)
		for layer in self.causal_conv_layers:
			output = layer(input_batch)
			input_batch = output
		return output

	def forward_residual_block(self, x_batch):
		params = self.params
		sum_skip_connections = 0
		input_batch = self.to_variable(x_batch)
		for i, block in enumerate(self.residual_blocks):
			for layer in block:
				output, z = layer(input_batch)
				sum_skip_connections += z
				input_batch = output

		return output, sum_skip_connections

	def forward_softmax_block(self, x_batch, softmax=True):
		input_batch = self.to_variable(x_batch)
		batchsize = self.get_batchsize(x_batch)
		for layer in self.softmax_conv_layers:
			input_batch = F.elu(input_batch)
			output = layer(input_batch)
			input_batch = output
		if softmax:
			output = F.softmax(output)
		return output

	# raw_network_output.ndim:	(batchsize, channels, 1, time_step)
	# target_signal_data.ndim:	(batchsize, time_step)
	def cross_entropy(self, raw_network_output, target_signal_data):
		if isinstance(target_signal_data, Variable):
			raise Exception("target_signal_data cannot be Variable")

		raw_network_output = self.to_variable(raw_network_output)
		target_width = target_signal_data.shape[1]
		batchsize = raw_network_output.data.shape[0]

		if raw_network_output.data.shape[3] != target_width:
			raise Exception("raw_network_output.width != target.width")

		# (batchsize * time_step,) <- (batchsize, time_step)
		target_signal_data = target_signal_data.reshape((-1,))
		target_signal = self.to_variable(target_signal_data)

		# (batchsize * time_step, channels) <- (batchsize, channels, 1, time_step)
		raw_network_output = F.transpose(raw_network_output, (0, 3, 2, 1))
		raw_network_output = F.reshape(raw_network_output, (batchsize * target_width, -1))

		loss = F.sum(F.softmax_cross_entropy(raw_network_output, target_signal))
		return loss

	def backprop(self, loss):
		self.zero_grads()
		loss.backward()
		self.update()

	def save(self, dir="./"):
		try:
			os.mkdir(dir)
		except:
			pass
		for i, layer in enumerate(self.causal_conv_layers):
			filename = dir + "/causal_conv_layer_{}.hdf5".format(i)
			serializers.save_hdf5(filename, layer)

		for i, block in enumerate(self.residual_blocks):
			for j, layer in enumerate(block):
				filename = dir + "/residual_{}_conv_layer_{}.hdf5".format(i, j)
				serializers.save_hdf5(filename, layer)

		for i, layer in enumerate(self.softmax_conv_layers):
			filename = dir + "/softmax_conv_layer_{}.hdf5".format(i)
			serializers.save_hdf5(filename, layer)
			

	def load(self, dir="./"):
		def load_hdf5(filename, layer):
			if os.path.isfile(filename):
				print "loading", filename
				serializers.load_hdf5(filename, layer)
			
		for i, layer in enumerate(self.causal_conv_layers):
			filename = dir + "/causal_conv_layer_{}.hdf5".format(i)
			load_hdf5(filename, layer)
			
		for i, block in enumerate(self.residual_blocks):
			for j, layer in enumerate(block):
				filename = dir + "/residual_{}_conv_layer_{}.hdf5".format(i, j)
				load_hdf5(filename, layer)
			
		for i, layer in enumerate(self.softmax_conv_layers):
			filename = dir + "/softmax_conv_layer_{}.hdf5".format(i)
			load_hdf5(filename, layer)



class DeepGenerativeModel(chainer.Chain):
	def __init__(self, **layers, params, n_layers):
		super(MultiLayerPerceptron, self).__init__(**layers)

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

		# Hidden
		for i in range(self.n_layers):
			u = chain[-1]
			if self.batchnorm_before_activation:
				u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				if self.apply_batchnorm and self.batchnorm_before_activation == False:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
			else:
				if self.apply_batchnorm:
					u = getattr(self, "batchnorm_%d" % i)(u, test=self.test)
			if self.batchnorm_before_activation == False:
				u = getattr(self, "layer_%i" % i)(u)
			if i == self.n_layers - 1:
				output = u
			else:
				output = f(u)
				if self.apply_dropout:
					output = F.dropout(output, train=not self.test)
			chain.append(output)

		return chain[-1]

	def __call__(self, x, test=False, apply_f=True):
		self.test = test
		output = self.compute_output(x)
		if apply_f:
			f = activations[self.activation_function]
			return f(output)
		return output