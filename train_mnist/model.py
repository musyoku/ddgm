# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from ddgm import DDGM, Params

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass
filename = args.model_dir + "/params.json"
if os.path.isfile(filename):
	print "loading", filename
	f = open(filename)
	try:
		dict = json.load(f)
		params = Params(dict)
	except:
		raise Exception("could not load {}".format(filename))

	params.gpu_enabled = True if args.gpu_enabled == 1 else False
else:
	params = Params()
	params.x_width = 28
	params.x_height = params.x_width
	params.ndim_x = params.x_width * params.x_width
	params.ndim_z = 10
	params.distribution_x = "sigmoid"

	params.energy_model_num_experts = 128
	params.energy_model_feature_extractor_hidden_units = [800, 800]
	params.energy_model_use_batchnorm = False
	# params.energy_model_batchnorm_to_input = True
	# params.energy_model_batchnorm_before_activation = False
	params.energy_model_use_dropout = True
	params.energy_model_use_weightnorm = True
	params.energy_model_weight_init_std = math.sqrt(0.02)
	params.energy_model_weight_initializer = "Normal"		# Normal or GlorotNormal
	params.energy_model_nonlinearity = "elu"
	params.energy_model_optimizer = "Adam"
	params.energy_model_learning_rate = 0.0002
	params.energy_model_momentum = 0.5
	params.energy_model_gradient_clipping = 10
	params.energy_model_weight_decay = 0

	params.generative_model_hidden_units = [500]
	params.generative_model_use_dropout = False
	params.generative_model_use_batchnorm = True
	params.generative_model_batchnorm_to_input = True
	params.generative_model_batchnorm_before_activation = True
	params.generative_model_use_weightnorm = False
	params.generative_model_weight_init_std = math.sqrt(0.02)
	params.generative_model_weight_initializer = "Normal"		# Normal or GlorotNormal
	params.generative_model_nonlinearity = "relu"
	params.generative_model_optimizer = "Adam"
	params.generative_model_learning_rate = 0.0002
	params.generative_model_momentum = 0.5
	params.generative_model_gradient_clipping = 10
	params.generative_model_weight_decay = 0

	params.gpu_enabled = True if args.gpu_enabled == 1 else False

	params.check()
	with open(filename, "w") as f:
		json.dump(params.to_dict(), f, indent=4)

params.dump()
ddgm = DDGM(params)
ddgm.load(args.model_dir)