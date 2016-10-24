# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from ddgm import DDGM, EnergyModelParams, GenerativeModelParams, DeepEnergyModel, DeepGenerativeModel
from sequential import Sequential
from sequential.link import Linear, BatchNormalization
from sequential.function import Activation, dropout, gaussian_noise, tanh, sigmoid

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
ndim_input = 2
ndim_latent_code = 4

# specify energy model
energy_model_filename = args.model_dir + "/energy_model.json"

if os.path.isfile(energy_model_filename):
	print "loading", energy_model_filename
	with open(energy_model_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(energy_model_filename))
else:
	config = EnergyModelParams()
	config.ndim_input = ndim_input
	config.num_experts = 4
	config.weight_init_std = 0.05
	config.weight_initializer = "Normal"
	config.use_weightnorm = True
	config.nonlinearity = "elu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	# feature extractor
	feature_extractor = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	feature_extractor.add(Linear(config.ndim_input, 128, use_weightnorm=config.use_weightnorm))
	feature_extractor.add(Activation(config.nonlinearity))
	feature_extractor.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	feature_extractor.add(tanh())
	feature_extractor.build()

	# experts
	experts = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	experts.add(Linear(128, config.num_experts, use_weightnorm=config.use_weightnorm))
	experts.build()

	# b
	b = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	b.add(Linear(config.ndim_input, 1, nobias=True))
	b.build()

	params = {
		"config": config.to_dict(),
		"feature_extractor": feature_extractor.to_dict(),
		"experts": experts.to_dict(),
		"b": b.to_dict(),
	}

	with open(energy_model_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

params_energy_model = params

# specify generative model
generative_model_filename = args.model_dir + "/generative_model.json"
if os.path.isfile(generative_model_filename):
	print "loading", generative_model_filename
	with open(generative_model_filename, "r") as f:
		try:
			params = json.load(f)
		except:
			raise Exception("could not load {}".format(generative_model_filename))
else:
	config = GenerativeModelParams()
	config.ndim_input = ndim_latent_code
	config.ndim_output = ndim_input
	config.distribution_output = "universal"
	config.use_weightnorm = False
	config.weight_init_std = 0.05
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0002
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	# model
	model = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	model.add(Linear(config.ndim_input, 128, use_weightnorm=config.use_weightnorm))
	model.add(BatchNormalization(128))
	model.add(Activation(config.nonlinearity))
	model.add(Linear(None, 128, use_weightnorm=config.use_weightnorm))
	model.add(BatchNormalization(128))
	model.add(Activation(config.nonlinearity))
	model.add(Linear(None, config.ndim_output, use_weightnorm=config.use_weightnorm))
	model.build()

	params = {
		"config": config.to_dict(),
		"model": model.to_dict(),
	}

	with open(generative_model_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

params_generative_model = params

ddgm = DDGM(params_energy_model, params_generative_model)
ddgm.load(args.model_dir)

if args.gpu_enabled == 1:
	ddgm.to_gpu()
