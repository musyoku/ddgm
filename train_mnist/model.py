# -*- coding: utf-8 -*-
import json, os, sys
from args import args
sys.path.append(os.path.split(os.getcwd())[0])
from ddgm import DDGM, Params

# load params.json
try:
	os.mkdir(args.params_dir)
except:
	pass
filename = args.params_dir + "/{}".format(args.params_filename)
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
	params.ndim_x = 28 * 28
	params.ndim_z = 10
	params.batchnorm_before_activation = True
	params.batchnorm_enabled = True
	params.activation_function = "elu"
	params.apply_dropout = False

	params.energy_model_num_experts = 128
	params.energy_model_features_hidden_units = [500]
	params.energy_model_apply_batchnorm_to_input = True

	params.generative_model_hidden_units = [500]
	params.generative_model_apply_batchnorm_to_input = False

	params.wscale = 0.1
	params.gradient_clipping = 0
	params.gradient_momentum = 0
	params.learning_rate = 0.001
	params.gpu_enabled = True if args.gpu_enabled == 1 else False

	with open(filename, "w") as f:
		params.check()
		json.dump(params.to_dict(), f, indent=4)

params.dump()
ddgm = DDGM(params)
ddgm.load(args.model_dir)