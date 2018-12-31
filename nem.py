import argparse
import numpy as np
import torch
import utils

from torch import distributions as dist

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_noise(data, noise_type=None, noise_prob=0.2):
	"""
	Add noise to the input image to avoid trivial solutions
	in case of overcapacity.
	"""
	if noise_type is None:
		return data

	else:
		shape = torch.stack([s.value if s.value is not None else data.size()[i]
                         for i, s in enumerate(data.size())])
		
		if noise_type == 'bitflip':
			noise_dist = dist.Bernoulli(probs=noise_prob)
			n = noise_dist.sample(shape)
			corrupted = data + n - 2 * data * n  # hacky way of implementing (data XOR n)
		else:
			raise KeyError('Unknown noise_type "{}"'.format(noise_type))

		corrupted.view(data.size())
		return corrupted


def nem_iterations():
	raise NotImplementedError


def main(log_dir, nr_steps, lr):
	if log_dir is None:
		utils.create_directory(log_dir)
		utils.clear_directory(log_dir)

	# set up data
	out_list = ['features', 'groups', 'collisions']

	train_inputs = PhysData(out_list=out_list)
	valid_inputs = PhysData(out_list=out_list)

	# build model
	# train_op, train_graph, valid_graph, debug_graph = build_graphs(train_inputs.output, valid_inputs.output)

	# training
	features_corrupted = add_noise(train_inputs['features'])
	loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, r_other_losses, \
    r_other_ub_losses = static_nem_iterations(features_corrupted, features,
                                              collisions=collisions, actions=actions)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default='./debug')
    parser.add_argument('--noise_type', type=str, default='bitflip')
    parser.add_argument('--noise_prob', type=float, default=0.2)
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--nr_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    config = parser.parse_args()
    print(config)
	main(config.log_dir, config.nr_steps, config.lr)