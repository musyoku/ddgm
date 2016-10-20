# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from ddgm import DDGM, EnergyModelParams, GenerativeModelParams

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
data = Params(name="data")
data.image_width = 28
data.image_height = data.image_width
data.ndim_latent_code = 10

# model parameters
energy_model_params_filename = args.model_dir + "/params_energy_model.json"
generative_model_params_filename = args.model_dir + "/params_generative_model.json"
params_energy_model = None
params_generative_model = None

# specify energy model
if os.path.isfile(energy_model_params_filename):
	print "loading", energy_model_params_filename
	with open(energy_model_params_filename, "w") as f:
		try:
			dict = json.load(f)
			params_energy_model = EnergyModelParams(dict)
		except:
			raise Exception("could not load {}".format(energy_model_params_filename))
else:
	params = EnergyModelParams(name="energy_model")
	params.ndim_input = data.image_width * data.image_height
	params.ndim_output = data.ndim_latent_code
	params.num_experts = 128
	params.feature_extractor_hidden_units = [800, 800]
	params.use_batchnorm = True
	params.batchnorm_to_input = True
	params.batchnorm_before_activation = False
	params.use_dropout_at_layer = [True, False]
	params.add_output_noise_at_layer = [True, True]
	params.use_weightnorm = True
	params.weight_init_std = 0.05
	params.weight_initializer = "GlorotNormal"
	params.nonlinearity = "elu"
	params.optimizer = "Adam"
	params.learning_rate = 0.0002
	params.momentum = 0.5
	params.gradient_clipping = 10
	params.weight_decay = 0

	params_energy_model = params
	params_energy_model.check()
	with open(energy_model_params_filename, "w") as f:
		json.dump(params_energy_model.to_dict(), f, indent=4)

# specify generative model
if os.path.isfile(generative_model_params_filename):
	print "loading", generative_model_params_filename
	with open(generative_model_params_filename, "w") as f:
		try:
			dict = json.load(f)
			params_generative_model = GenerativeModelParams(dict)
		except:
			raise Exception("could not load {}".format(generative_model_params_filename))
else:
	params = GenerativeModelParams(name="generative_model")
	params.ndim_input = data.ndim_latent_code
	params.ndim_output = data.image_width * data.image_height
	params.distribution_output = "sigmoid"
	params.hidden_units = [800, 800]
	params.use_dropout_at_layer = [False, False]
	params.add_output_noise_at_layer = [False, False]
	params.use_batchnorm = True
	params.batchnorm_to_input = True
	params.batchnorm_before_activation = True
	params.use_weightnorm = False
	params.weight_init_std = 1
	params.weight_initializer = "GlorotNormal"
	params.nonlinearity = "relu"
	params.optimizer = "Adam"
	params.learning_rate = 0.0002
	params.momentum = 0.5
	params.gradient_clipping = 10
	params.weight_decay = 0

	params_generative_model = params
	params_generative_model.check()
	with open(energy_model_params_filename, "w") as f:
		json.dump(params_generative_model.to_dict(), f, indent=4)

data.dump()
params_energy_model.dump()
params_generative_model.dump()
ddgm = DDGM(params_energy_model, params_generative_model)
ddgm.load(args.model_dir)
params = ddgm.params