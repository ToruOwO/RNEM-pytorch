import argparse
import numpy as np
import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

from data import Data
from model import InnerConvAE
from nem import NEM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = None

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


def compute_outer_loss(mu, gamma, target, prior, collision):
	# use binomial cross entropy as intra loss
	intra_criterion = nn.BCELoss()

	# use KL divergence as inter loss
	inter_criterion = nn.KLDivLoss()

	intra_loss = intra_criterion(mu, target)
	inter_loss = inter_criterion(prior, mu)

	batch_size = target.size()[0]

	# compute rel losses
	r_intra_loss = torch.sum(collision * intra_loss * gamma.detach()) / batch_size
	r_inter_loss = torch.sum(collision * inter_loss * (1. - gamma.detach())) / batch_size

	# compute normal losses
	intra_loss = torch.sum(intra_loss * gamma.detach()) / batch_size
	inter_loss = torch.sum(inter_loss * (1.0 - gamma.detach())) / batch_size

	total_loss = intra_loss + inter_loss
	r_total_loss = r_intra_loss + r_inter_loss

	return total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss


def compute_outer_ub_loss(pred, target, prior, collision):
	max_pred = torch.max(pred, dim=1, keepdim=True)

	# use binomial cross entropy as intra loss
	intra_criterion = nn.BCELoss()

	# use KL divergence as inter loss
	inter_criterion = nn.KLDivLoss()

	intra_ub_loss = intra_criterion(max_pred, target)
	inter_ub_loss = inter_criterion(prior, max_pred)

	batch_size = target.size()[0]

	r_intra_ub_loss = torch.sum(collision * intra_ub_loss) / batch_size
	r_inter_ub_loss = torch.sum(collision * inter_ub_loss) / batch_size

	intra_ub_loss = torch.sum(intra_ub_loss) / batch_size
	inter_ub_loss = torch.sum(inter_ub_loss) / batch_size

	total_ub_loss = intra_ub_loss + inter_ub_loss
	r_total_ub_loss = r_intra_ub_loss + r_inter_ub_loss

	return total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss


def nem_iterations(input_data, target_data, collisions=None, k=5, is_training=True):
	# get input dimensions
	input_shape = input_data.size()

	print("input data is of size", input_shape)

	assert input_shape[0] == 6, "Requires 6D input (T, B, K, W, H, C) but {}".format(input_shape[0])
	W, H, C = (x for x in input_shape[-3:])

	# set initial distribution (Bernoulli) of pixels
	inner_model = InnerConvAE(K=k)
	nem_model = NEM(inner_model, (W, H, C), inner_model.hidden_size, inner_model.input_size)

	# compute Bernoulli prior
	prior = compute_bernoulli_prior()

	# use binomial cross entropy as intra loss
	intra_criterion = nn.BCELoss()

	# use KL divergence as inter loss
	inter_criterion = nn.KLDivLoss()

	# use Adam optimizer
	optimizer = optim.Adam(nem_model.parameters(), lr=args.lr)

	# outputs
	hidden_state = nem_model.init_state(input_shape[1], k)

	# record losses
	total_losses = []
	total_ub_losses = []
	r_total_losses = []
	r_total_ub_losses = []
	other_losses = []
	other_ub_losses = []
	r_other_losses = []
	r_other_ub_losses = []

	outputs = [hidden_state]

	if is_training:
		model.train()

	losses = 0.0

	for t in range(args.nr_steps):
		# model should predict the next frame
		inputs = (input_data[t], target_data[t+1])

		# forward pass
		hidden_state, output = nem_model(inputs, hidden_state)
		theta, pred, gamma = output

		# use collision data
		collision = torch.zeros(1, 1, 1, 1, 1) if collisions is None else collisions[t]

		# compute NEM losses
		total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss \
			= compute_outer_loss(pred, gamma, target_data[t+1], prior, collision=collision)

		total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss \
			= compute_outer_ub_loss(pred, target_data[t+1], prior, collision=collision)

		total_loss = intra_criterion(output, labels) + inter_criterion(output, labels)
		losses += total_loss

		total_losses.append(total_loss)
		total_ub_losses.append(total_ub_loss)

		r_total_losses.append(r_total_loss)
		r_total_ub_losses.append(r_total_ub_loss)

		other_losses.append(torch.stack((total_loss, intra_loss, inter_loss)))
		other_ub_losses.append(torch.stack((total_ub_loss, intra_ub_loss, inter_ub_loss)))


		r_other_losses.append(torch.stack((r_total_loss, r_intra_loss, r_inter_loss)))
		r_other_ub_losses.append(torch.stack((r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss)))

		# backward pass and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print log
		if (i+1) % 100 == 0:
			print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

	losses /= len(input_data)
	print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
          (phase, epoch, args.n_epoch, losses, best_valid_loss))

	thetas, preds, gammas = zip(*outputs)
	thetas = torch.stack((thetas,))
	preds = torch.stack((preds,))
	gammas = torch.stack((gammas,))

	return loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
		 r_other_losses, r_other_ub_losses


def main():
	if args.log_dir is None:
		utils.create_directory(log_dir)
		utils.clear_directory(log_dir)

	# set up input data
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + 1

	train_inputs = Data(args.data_name, 'training', sequence_length=nr_iters, attribute_list=attribute_list)
	valid_inputs = Data(args.data_name, 'validation', sequence_length=nr_iters, attribute_list=attribute_list)

	# build model
	model = InnerConvAE()

	# training
	best_valid_loss = np.inf
	best_valid_epoch = 0

	for epoch in range(1, args.max_epoch + 1):
		# training phase
		features_corrupted = add_noise(train_inputs['features'], noise_type=args.noise_type)
		features = train_inputs['features']

		# TODO: convert into a log dict
		loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
		r_other_losses, r_other_ub_losses = nem_iterations(features_corrupted, 
													    	features, 
													    	collision=train_inputs.get('collisions', None))

		# validation phase
		features_corrupted_valid = add_noise(valid_inputs['features'], noise_type=args.noise_type)
		features_valid = valid_inputs['features']


		loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
		r_other_losses, r_other_ub_losses = nem_iterations(features_corrupted_valid, 
													    	features_valid, 
													    	collision=train_inputs.get('collisions', None))

		if loss < best_valid_loss:
			best_valid_loss = loss
			best_valid_epoch = epoch
			print("    Best validation loss improved to %.03f" % best_valid_loss)
			torch.save(model.state_dict(), os.path.abspath(os.path.join(log_dir, 'best.pth')))
			print("    Saved to:", save_destination)

		if epoch % log_per_iter == 0:
			torch.save(model.state_dict(), os.path.abspath(os.path.join(log_dir, 'epoch_{}.pth'.format(epoch))))

		if np.isnan(loss):
			print('Early Stopping because validation loss is nan')
			break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name', type=str, default='balls3curtain64')
	parser.add_argument('--log_dir', type=str, default='./debug')
	parser.add_argument('--nr_steps', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--max_epoch', type=int, default=500)
	parser.add_argument('--noise_type', type=str, default='bitflip')
	parser.add_argument('--log_per_iter', type=int, default=50)

	args = parser.parse_args()
	print(args)

	main()