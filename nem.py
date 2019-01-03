import argparse
import numpy as np
import utils
import torch
import torch.optim as optim
import torch.distributions as dist

from model import InnerConvAE

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


def compute_bernoulli_prior():
	"""
	Compute Bernoulli prior over the input data with p = 0.0
	"""
	return torch.zeros(1, 1, 1, 1, 1)


def nem_iterations(input_data, target_data, k, num_epochs, learning_rate=0.001):
	# get input dimensions
	input_shape = input_data.size()
	assert input_shape[0] == 6, "Requires 6D input (T, B, K, W, H, C) but {}".format(input_shape[0])
	W, H, C = (x for x in input_shape[-3:])

	# set initial distribution (Bernoulli) of pixels
	inner_model = InnerConvAE(K=k)
	nem_model = NEMCell(inner_cell, input_shape=(W, H, C))

	# compute Bernoulli prior
	prior = compute_bernoulli_prior()

	# use binomial cross entropy as intra loss
	intra_criterion = nn.BCELoss()

	# use KL divergence as inter loss
	inter_criterion = nn.KLDivLoss()

	# use Adam optimizer
	optimizer = optim.Adam(nem_model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		# forward pass
		outputs = nem_model(data)
		total_loss = intra_criterion(outputs, labels) + inter_criterion(outputs, labels)

		# backward pass and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print log
		if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))

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