import argparse
import numpy as np
import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.utils.data

from data import Data
from nem import NEM
from utils import BCELoss, KLDivLoss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = None

def add_noise(data, noise_type=None, noise_prob=0.2):
	"""
	Add noise to the input image to avoid trivial solutions
	in case of overcapacity.

	shape of returned data: (B, K, W, H, C)
	"""
	if noise_type is None:
		return data

	else:
		shape = data[0].size()
		corrupted_data = []

		for i in range(len(data)):

			if noise_type == 'bitflip':
				noise_dist = dist.Bernoulli(probs=noise_prob)
				n = noise_dist.sample(shape)
				corrupted = data[i] + n - 2 * data[i] * n  # hacky way of implementing (data XOR n)
			else:
				raise KeyError('Unknown noise_type "{}"'.format(noise_type))

			corrupted.view(shape)
			corrupted_data.append(corrupted)

		corrupted_data = torch.stack(corrupted_data)
		# print(corrupted_data.size())

		return corrupted_data


def compute_bernoulli_prior():
	"""
	Compute Bernoulli prior over the input data with p = 0.0
	"""
	return torch.zeros(1, 1, 1, 1, 1)


def compute_outer_loss(mu, gamma, target, prior, collision):
	# use binomial cross entropy as intra loss
	intra_criterion = BCELoss()

	# use KL divergence as inter loss
	inter_criterion = KLDivLoss()

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
	max_pred, _ = torch.max(pred, dim=1)
	max_pred = torch.unsqueeze(max_pred, 1)

	# use binomial cross entropy as intra loss
	intra_criterion = nn.BCELoss()

	# use KL divergence as inter loss
	inter_criterion = KLDivLoss()

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


def dynamic_nem_iterations(input_data, target_data, h_old, preds_old, gamma_old, collisions=None):
	# get input dimensions
	input_shape = input_data.size()

	print("input data is of size", input_shape)

	assert len(input_shape) == 5, "Requires 5D input (B, K, W, H, C)"
	W, H, C = (x for x in input_shape[-3:])

	# set up NEM model
	nem_model = NEM(batch_size=input_shape[1], k=args.k, input_size=(W, H, C), hidden_size=args.inner_hidden_size).to(device)

	saved_model_path = os.path.join(args.save_dir, args.saved_model)

	nem_model.load_state_dict(torch.load(saved_model_path))
	nem_model.eval()

	# compute Bernoulli prior of pixels
	prior = compute_bernoulli_prior()

	# compute inputs for dynamic iterations
	inputs = (input_data, target_data)
	hidden_state = (h_old, preds_old, gamma_old)

	# run hidden network
	hidden_state, output = nem_model(inputs, hidden_state)
	theta, pred, gamma = output

	# set collision
	collision = torch.zeros(1, 1, 1, 1, 1) if collisions is None else collisions[t]

	# compute NEM losses
	total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss \
		= compute_outer_loss(pred, gamma, target_data[t + 1], prior, collision=collision)

	# compute estimated loss upper bound (which doesn't use E-step)
	total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss \
		= compute_outer_ub_loss(pred, target_data[t + 1], prior, collision=collision)

	other_losses = torch.stack((total_loss, intra_loss, inter_loss))
	other_ub_losses = torch.stack((total_ub_loss, intra_ub_loss, inter_ub_loss))

	r_other_losses = torch.stack((r_total_loss, r_intra_loss, r_inter_loss))
	r_other_ub_losses = torch.stack((r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss))

	return total_loss, total_ub_loss, r_total_loss, r_total_ub_loss, theta, pred, gamma, other_losses, \
		   other_ub_losses, r_other_losses, r_other_ub_losses


def nem_iterations(input_data, target_data, collisions=None, is_training=True):
	# get input dimensions
	input_shape = input_data.size()

	print("input data is of size", input_shape)

	assert len(input_shape) == 6, "Requires 6D input (T, B, K, W, H, C)"
	W, H, C = (x for x in input_shape[-3:])

	# set up initial inner RNN and NEM model
	nem_model = NEM(batch_size=input_shape[1], k=args.k, input_size=(W, H, C), hidden_size=args.inner_hidden_size).to(device)

	if args.saved_model != None or args.saved_model != "":
		# set up trained NEM model
		saved_model_path = os.path.join(args.save_dir, args.saved_model)
		nem_model.load_state_dict(torch.load(saved_model_path))

	# compute Bernoulli prior of pixels
	prior = compute_bernoulli_prior()

	# outputs
	hidden_state = nem_model.hidden_state

	# use Adam optimizer
	optimizer = optim.Adam(list(nem_model.parameters()) + list(nem_model.inner_rnn.parameters()), lr=args.lr)

	# record losses
	total_losses = []
	total_ub_losses = []
	r_total_losses = []
	r_total_ub_losses = []
	other_losses = []
	other_ub_losses = []
	r_other_losses = []
	r_other_ub_losses = []

	loss_step_weights = [1.0] * args.nr_steps

	outputs = [hidden_state]

	if is_training:
		nem_model.train()
	else:
		nem_model.eval()

	losses = 0.0

	for t, loss_weight in enumerate(loss_step_weights):
		# model should predict the next frame
		inputs = (input_data[t], target_data[t+1])

		assert len(input_data[t]) == len(target_data[t + 1]), \
			"Input data and target data must have the same shape"

		# print("inputs", inputs, inputs[0].size(), inputs[1].size())

		# forward pass
		hidden_state, output = nem_model(inputs, hidden_state)
		theta, pred, gamma = output

		# use collision data
		collision = torch.zeros(1, 1, 1, 1, 1) if collisions is None else collisions[t]

		# compute NEM losses
		total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss \
			= compute_outer_loss(pred, gamma, target_data[t+1], prior, collision=collision)

		# compute estimated loss upper bound (which doesn't use E-step)
		total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss \
			= compute_outer_ub_loss(pred, target_data[t+1], prior, collision=collision)

		total_losses.append(total_loss)
		total_ub_losses.append(total_ub_loss)

		r_total_losses.append(r_total_loss)
		r_total_ub_losses.append(r_total_ub_loss)

		other_losses.append(torch.stack((total_loss, intra_loss, inter_loss)))
		other_ub_losses.append(torch.stack((total_ub_loss, intra_ub_loss, inter_ub_loss)))

		r_other_losses.append(torch.stack((r_total_loss, r_intra_loss, r_inter_loss)))
		r_other_ub_losses.append(torch.stack((r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss)))

		outputs.append(output)   # thetas, preds, gammas

		if t % args.step_log_per_iter == 0:
			print("Step [{}/{}], Loss: {:.4f}".format(t, args.nr_steps, total_loss))

		# backward pass and optimize
		optimizer.zero_grad()
		total_loss.backward(retain_graph=True)
		optimizer.step()

	# collect outputs
	thetas, preds, gammas = zip(*outputs)
	thetas = torch.stack(thetas)
	preds = torch.stack(preds)
	gammas = torch.stack(gammas)

	other_losses = torch.stack(other_losses)
	other_ub_losses = torch.stack(other_ub_losses)
	r_other_losses = torch.stack(r_other_losses)
	r_other_ub_losses = torch.stack(r_other_ub_losses)

	total_loss = torch.sum(torch.stack(total_losses)) / np.sum(loss_step_weights)
	total_ub_loss = torch.sum(torch.stack(total_ub_losses)) / np.sum(loss_step_weights)
	r_total_loss = torch.sum(torch.stack(r_total_losses)) / np.sum(loss_step_weights)
	r_total_ub_loss = torch.sum(torch.stack(r_total_ub_losses)) / np.sum(loss_step_weights)

	return total_loss, total_ub_loss, r_total_loss, r_total_ub_loss, thetas, preds, gammas, other_losses, \
		other_ub_losses, r_other_losses, r_other_ub_losses, nem_model


def rollout_from_file():
	# set up input data
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + args.rollout_steps + 1

	input_data = {
		attribute: Data(args.data_name,
						'test',
						batch_id=0,
						sequence_length=nr_iters,
						attribute=attribute) for attribute in attribute_list
	}

	# initialize RNN hidden state, prediction and gamma
	theta = torch.zeros(args.batch_size * args.k, 250)
	pred = torch.ones(args.batch_size, args.k, 64, 64, 1)                  # (B, K, W, H, C)
	gamma = np.abs(np.random.randn(args.batch_size, args.k, 64, 64, 1))    # (B, K, W, H, 1)
	gamma /= np.sum(gamma, axis=1, keepdims=True)
	gamma = torch.from_numpy(gamma)

	corrupted, scores, gammas, thetas, preds = [], [], [gamma], [theta], [pred]
	
	# run rollout steps
	for t in range(nr_iters - 1):
		if 'collisions' in input_data:
			collisions = input_data['collisions'][t]
		else:
			collisions = None

		corr = add_noise(input_data['features'][t])

		loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, \
		r_other_losses, r_other_ub_losses = dynamic_nem_iterations(input_data=corr,
																	target_data=input_data['features'][t+1],
																	gamma_old=gamma,
																	h_old=theta,
																	preds_old=pred,
																	collisions=collisions)

		# re-compute gamma if rollout
		if t >= args.nr_steps:
			truth = torch.max(preds, dim=1, keepdims=True)

			# avoid vanishing by scaling or sampling
			truth[truth > 0.1] = 1.0
			truth[truth <= 0.1] = 0.0

			# compute probs
			probs = truth * pred + (1 - truth) * (1 - pred)

			# add epsilon to probs in order to prevent 0 gamma
			probs += 1e-6

			# compute the new gamma (E-step) or set to one for k=1
			gamma = probs / torch.sum(probs, 1, keepdims=True) if args.k > 1 else torch.ones_like(gamma)

		corrupted.append(corr)
		gammas.append(gamma)
		thetas.append(theta)
		preds.append(pred)


def print_log_dict(usage, loss, ub_loss, r_loss, r_ub_loss, other_losses, other_ub_losses, r_other_losses, \
					r_other_ub_losses, loss_step_weights):
	dt = args.dt
	s_loss_weights = np.sum(loss_step_weights)
	dt_s_loss_weights = np.sum(loss_step_weights[-dt:])

	print("%s Loss: %.3f (UB: %.3f), Relational Loss: %.3f (UB: %.3f)" % (usage, loss, ub_loss, r_loss, r_ub_loss))

	print("    other losses: {}".format(", ".join(["%.2f (UB: %.2f)" % 
																(other_losses[:, i].sum(0) / s_loss_weights,
																other_ub_losses[:, i].sum(0) / s_loss_weights)
																for i in range(len(other_losses[0]))])))

	print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
																(other_losses[-dt:, i].sum(0) / dt_s_loss_weights,
																other_ub_losses[-dt:, i].sum(0) / dt_s_loss_weights)
																for i in range(len(other_losses[0]))])))

	print("    other relational losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
																(r_other_losses[:, i].sum(0) / s_loss_weights,
																	r_other_ub_losses[:, i].sum(0) / s_loss_weights)
																for i in range(len(r_other_losses[0]))])))

	print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
																(r_other_losses[-dt:, i].sum(0) / dt_s_loss_weights,
																r_other_ub_losses[-dt:, i].sum(0) / dt_s_loss_weights)
																for i in range(len(r_other_losses[0]))])))


def run_from_file():
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + 1
	loss_step_weights = [1.0] * args.nr_steps

	# TODO: record data using dictionary
	for epoch in range(1, args.max_epoch + 1):
		print("Starting epoch {}...".format(epoch))

		for b in range(Data.get_num_batches()):
			inputs = {
				attribute: Data(args.data_name, 'test', b, sequence_length=nr_iters, attribute=attribute)
				for attribute in attribute_list
			}

			# training phase
			features_corrupted = add_noise(inputs['features'], noise_type=args.noise_type)
			features = inputs['features']

			# TODO: convert into a log dict
			loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
			r_other_losses, r_other_ub_losses, train_model = nem_iterations(features_corrupted,
															   features,
															   collisions=inputs.get('collisions', None))

			print_log_dict(loss, ub_loss, r_loss, r_ub_loss, other_losses, other_ub_losses, r_other_losses, \
				r_other_ub_losses, loss_step_weights)


def run():
	log_dir = args.log_dir

	utils.create_directory(log_dir)
	utils.clear_directory(log_dir)

	# set up input data
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + 1

	# training
	best_valid_loss = np.inf
	best_valid_epoch = 0

	for epoch in range(1, args.max_epoch + 1):
		print("Starting epoch {}...".format(epoch))

		for b in range(Data.get_num_batches()):
			train_inputs = {
				attribute: Data(args.data_name, 'training', b, sequence_length=nr_iters, attribute=attribute)
				for attribute in attribute_list
			}
			valid_inputs = {
				attribute: Data(args.data_name, 'validation', b, sequence_length=nr_iters, attribute=attribute)
				for attribute in attribute_list
			}

			# training phase
			features_corrupted = add_noise(train_inputs['features'], noise_type=args.noise_type)
			features = train_inputs['features']

			# TODO: convert into a log dict
			loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
			r_other_losses, r_other_ub_losses, train_model = nem_iterations(features_corrupted,
															   features,
															   collisions=train_inputs.get('collisions', None))


			# validation phase
			features_corrupted_valid = add_noise(valid_inputs['features'], noise_type=args.noise_type)
			features_valid = valid_inputs['features']


			loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
			r_other_losses, r_other_ub_losses, valid_model = nem_iterations(features_corrupted_valid,
															   features_valid,
															   collisions=valid_inputs.get('collisions', None))

			if loss < best_valid_loss:
				best_valid_loss = loss
				best_valid_epoch = epoch
				print("Best validation loss improved to %.03f" % best_valid_loss)
				print("Best valid epoch [{}/{}]".format(best_valid_epoch, args.max_epoch + 1))
				torch.save(train_model.state_dict(), os.path.abspath(os.path.join(log_dir, 'best.pth')))
				print("===Saved to:", args.save_dir)

			if epoch % args.log_per_iter == 0:
				print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, args.max_epoch + 1, loss))
				torch.save(train_model.state_dict(), os.path.abspath(os.path.join(log_dir, 'epoch_{}.pth'.format(epoch))))

			if np.isnan(loss.detach()):
				print("Early Stopping because validation loss is nan")
				break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name', type=str, default='balls3curtain64')
	parser.add_argument('--log_dir', type=str, default='./debug')
	parser.add_argument('--save_dir', type=str, default='./trained_model')
	parser.add_argument('--nr_steps', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--max_epoch', type=int, default=500)
	parser.add_argument('--dt', type=int, default=10)
	parser.add_argument('--noise_type', type=str, default='bitflip')
	parser.add_argument('--log_per_iter', type=int, default=10)
	parser.add_argument('--step_log_per_iter', type=int, default=10)
	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--data_batch_size', type=int, default=10)
	parser.add_argument('--inner_hidden_size', type=int, default=250)
	parser.add_argument('--saved_model', type=str, default='best.pth')
	parser.add_argument('--rollout_steps', type=int, default=10)
	parser.add_argument('--eval', type=bool, default=False)

	args = parser.parse_args()
	print("=== Arguments ===")
	print(args)
	print()

	if args.eval:
		run_from_file()
	else:
		run()
