import os

import argparse
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import utils
from data import Data
from nem import NEM
from utils import BCELoss, KLDivLoss

# Device configuration
use_gpu = None
device = None

args = None


### helper functions

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
				n = noise_dist.sample(shape).to(device)
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
	# convert to cuda tensor on GPU
	return torch.zeros(1, 1, 1, 1, 1).to(device)


def compute_outer_loss(mu, gamma, target, prior, collision):
	# use binomial cross entropy as intra loss
	intra_criterion = BCELoss().to(device)

	# use KL divergence as inter loss
	inter_criterion = KLDivLoss().to(device)

	intra_loss = intra_criterion(mu, target, use_gpu=use_gpu)
	inter_loss = inter_criterion(prior, mu, use_gpu=use_gpu)

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
	intra_criterion = nn.BCELoss().to(device)

	# use KL divergence as inter loss
	inter_criterion = KLDivLoss().to(device)

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


### run epoch iterations


def dynamic_nem_iterations(input_data, target_data, h_old, preds_old, gamma_old, nem_model, collisions=None):
	# get input dimensions
	input_shape = input_data.size()

	print("input data is of size", input_shape)

	assert len(input_shape) == 5, "Requires 5D input (B, K, W, H, C)"
	W, H, C = (x for x in input_shape[-3:])
	assert (W, H, C) == nem_model.input_size, "Require NEM input size to be (W, H, C)"

	# evaluation mode
	nem_model.eval()

	# compute Bernoulli prior of pixels
	# convert to cuda tensor on GPU
	prior = compute_bernoulli_prior()

	# compute inputs for dynamic iterations
	inputs = (input_data, target_data)
	hidden_state = (h_old, preds_old, gamma_old)

	# run hidden network
	hidden_state, output = nem_model(inputs, hidden_state)
	theta, pred, gamma = output

	# set collision
	collision = torch.zeros(1, 1, 1, 1, 1).to(device) if collisions is None else collisions

	# compute NEM losses
	total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss \
		= compute_outer_loss(pred, gamma, target_data, prior, collision=collision)

	# compute estimated loss upper bound (which doesn't use E-step)
	total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss \
		= compute_outer_ub_loss(pred, target_data, prior, collision=collision)

	other_losses = torch.stack((total_loss, intra_loss, inter_loss))
	other_ub_losses = torch.stack((total_ub_loss, intra_ub_loss, inter_ub_loss))

	r_other_losses = torch.stack((r_total_loss, r_intra_loss, r_inter_loss))
	r_other_ub_losses = torch.stack((r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss))

	return total_loss, total_ub_loss, r_total_loss, r_total_ub_loss, theta, pred, gamma, other_losses, \
	       other_ub_losses, r_other_losses, r_other_ub_losses


def nem_iterations(input_data, target_data, nem_model, collisions=None, is_training=True):
	# get input dimensions
	input_shape = input_data.size()

	print("input data is of size", input_shape)

	assert len(input_shape) == 6, "Requires 6D input (T, B, K, W, H, C)"
	W, H, C = (x for x in input_shape[-3:])
	assert (W, H, C) == nem_model.input_size, "Require NEM input size to be (W, H, C)"

	# compute Bernoulli prior of pixels
	prior = compute_bernoulli_prior()

	# output
	hidden_state = (nem_model.h, nem_model.pred, nem_model.gamma)

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

	for t, loss_weight in enumerate(loss_step_weights):
		# model should predict the next frame
		inputs = (input_data[t], target_data[t + 1])

		assert len(input_data[t]) == len(target_data[t + 1]), \
			"Input data and target data must have the same shape"

		# print("inputs", inputs, inputs[0].size(), inputs[1].size())

		# forward pass
		hidden_state, output = nem_model(inputs, hidden_state)
		theta, pred, gamma = output

		# use collision data
		collision = torch.zeros(1, 1, 1, 1, 1).to(device) if collisions is None else collisions[t]

		# compute NEM losses
		total_loss, intra_loss, inter_loss, r_total_loss, r_intra_loss, r_inter_loss \
			= compute_outer_loss(pred, gamma, target_data[t + 1], prior, collision=collision)

		# compute estimated loss upper bound (which doesn't use E-step)
		total_ub_loss, intra_ub_loss, inter_ub_loss, r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss \
			= compute_outer_ub_loss(pred, target_data[t + 1], prior, collision=collision)

		total_losses.append(total_loss)
		total_ub_losses.append(total_ub_loss)

		r_total_losses.append(r_total_loss)
		r_total_ub_losses.append(r_total_ub_loss)

		other_losses.append(torch.stack((total_loss, intra_loss, inter_loss)))
		other_ub_losses.append(torch.stack((total_ub_loss, intra_ub_loss, inter_ub_loss)))

		r_other_losses.append(torch.stack((r_total_loss, r_intra_loss, r_inter_loss)))
		r_other_ub_losses.append(torch.stack((r_total_ub_loss, r_intra_ub_loss, r_inter_ub_loss)))

		outputs.append(output)  # thetas, preds, gammas

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


### log computation results


def print_log_dict(loss, ub_loss, r_loss, r_ub_loss, other_losses, other_ub_losses, r_other_losses,
                   r_other_ub_losses, loss_step_weights):
	dt = args.dt
	s_loss_weights = np.sum(loss_step_weights)
	dt_s_loss_weights = np.sum(loss_step_weights[-dt:])

	print("Loss: %.3f (UB: %.3f), Relational Loss: %.3f (UB: %.3f)" % (loss, ub_loss, r_loss, r_ub_loss))

	# print("    other losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
	#                                                (other_losses[:, i].sum(0) / s_loss_weights,
	#                                                 other_ub_losses[:, i].sum(0) / s_loss_weights)
	#                                                for i in range(len(other_losses[0]))])))
	#
	# print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
	#                                                             (other_losses[-dt:, i].sum(0) / dt_s_loss_weights,
	#                                                              other_ub_losses[-dt:, i].sum(0) / dt_s_loss_weights)
	#                                                             for i in range(len(other_losses[0]))])))
	#
	# print("    other relational losses: {}".format(", ".join(["%.2f (UB: %.2f)" %
	#                                                           (r_other_losses[:, i].sum(0) / s_loss_weights,
	#                                                            r_other_ub_losses[:, i].sum(0) / s_loss_weights)
	#                                                           for i in range(len(r_other_losses[0]))])))
	#
	# print("        last {} steps avg: {}".format(dt, ", ".join(["%.2f (UB: %.2f)" %
	#                                                             (r_other_losses[-dt:, i].sum(0) / dt_s_loss_weights,
	#                                                              r_other_ub_losses[-dt:, i].sum(0) / dt_s_loss_weights)
	#                                                             for i in range(len(r_other_losses[0]))])))


def create_rollout_plots(name, outputs, idx):
	import matplotlib.pyplot as plt
	scores, confidences = len(idx) * [0.0], len(idx) * [0.0]

	# produce overview plot
	for i, nr in enumerate(idx):
		fig = utils.overview_plot(i, **outputs)
		fig.suptitle(name + ', sample {},  AMI Score: {:.3f} ({:.3f}) '.format(nr, scores[i], confidences[i]))
		fig.savefig(os.path.join(args.log_dir, name + '_{}.png'.format(nr)), bbox_inches='tight', pad_inches=0)
		plt.close(fig)

		utils.overview_gif('rollout', nr, args.nr_steps, args.rollout_steps, **outputs)


### Main functions


def rollout_from_file():
	# set up input data
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + args.rollout_steps + 1

	# set up model
	model = NEM(batch_size=args.batch_size,
	            k=args.k,
	            input_size=(64, 64, 1),
	            hidden_size=args.inner_hidden_size,
	            device=device).to(device)

	# a model must be provided in order to rollout from file
	assert args.saved_model != None and args.saved_model != "", "Please provide a pre-trained model"
	saved_model_path = os.path.join(args.save_dir, args.saved_model)
	assert os.path.isfile(saved_model_path), "Path to model does not exist"
	model.load_state_dict(torch.load(saved_model_path))

	# create empty lists to record losses
	losses, ub_losses, r_losses, r_ub_losses, others, others_ub, r_others, r_others_ub = [], [], [], [], [], [], [], []

	loss_step_weights = [1.0] * args.nr_steps

	for b in range(Data.get_num_batches()):
		input_data = {
			attribute: Data(args.data_name,
			                'test',
			                batch_id=b,
			                sequence_length=nr_iters,
			                attribute=attribute) for attribute in attribute_list
		}

		# convert numpy bool array to tensor on GPU
		for k, v in input_data.items():
			input_data[k] = torch.from_numpy(v.data.astype(float)).float().to(device)

		# initialize RNN hidden state, prediction and gamma
		theta = torch.zeros(args.batch_size * args.k, 250)
		pred = torch.ones(args.batch_size, args.k, 64, 64, 1)  # (B, K, W, H, C)
		gamma = np.abs(np.random.randn(args.batch_size, args.k, 64, 64, 1))  # (B, K, W, H, 1)
		gamma /= np.sum(gamma, axis=1, keepdims=True)
		gamma = torch.from_numpy(gamma).float()

		corrupted, scores, gammas, thetas, preds = [], [], [gamma], [theta], [pred]

		# record losses
		losses.append([])
		ub_losses.append([])
		r_losses.append([])
		r_ub_losses.append([])
		others.append([])
		others_ub.append([])
		r_others.append([])
		r_others_ub.append([])

		# run rollout steps
		for t in range(nr_iters - 1):
			if 'collisions' in input_data:
				collisions = input_data['collisions'][t]
			else:
				collisions = None

			# decide if the model is rolling out or using real data
			if t < args.nr_steps:
				# real data
				input = input_data['features'][t]
			else:
				# rollout
				input = torch.sum(gamma * pred, 1, keepdim=True)

			# run forward process
			input_corrupted = add_noise(input)
			loss, ub_loss, r_loss, r_ub_loss, theta, pred, gamma, other_losses, other_ub_losses, \
			r_other_losses, r_other_ub_losses = dynamic_nem_iterations(input_data=input_corrupted,
			                                                           target_data=input_data['features'][t + 1],
			                                                           gamma_old=gamma,
			                                                           h_old=theta,
			                                                           preds_old=pred,
			                                                           nem_model=model,
			                                                           collisions=collisions)

			# re-compute gamma if rollout
			if t >= args.nr_steps:
				# torch.max returns two values, where the second value is argmax
				truth, _ = torch.max(pred, 1, keepdim=True)

				# avoid vanishing by scaling or sampling
				ones = torch.ones_like(truth)
				zeros = torch.zeros_like(truth)
				truth = torch.where(truth > 0.1, truth, ones)
				truth = torch.where(truth <= 0.1, truth, zeros)

				# compute probs
				probs = truth * pred + (1 - truth) * (1 - pred)

				# add epsilon to probs in order to prevent 0 gamma
				probs += 1e-6

				# compute the new gamma (E-step) or set to one for k=1
				gamma = probs / torch.sum(probs, 1, keepdim=True) if args.k > 1 else torch.ones_like(gamma)

			corrupted.append(input_corrupted)
			gammas.append(gamma)
			thetas.append(theta)
			preds.append(pred)

			losses[-1].append(loss)
			ub_losses[-1].append(ub_loss)
			r_losses[-1].append(r_loss)
			r_ub_losses[-1].append(r_ub_loss)
			others[-1].append(other_losses)
			others_ub[-1].append(other_ub_losses)
			r_others[-1].append(r_other_losses)
			r_others_ub[-1].append(r_other_ub_losses)

		# collect outputs for graph drawing
		outputs = {
			'inputs': input_data['features'],
			'corrupted': torch.stack(corrupted),
			'gammas': torch.stack(gammas),
			'preds': torch.stack(preds),
		}

		if b == 0:
			idx = [0, 1, 2]   # sample ids to generate plots
			create_rollout_plots('rollout', outputs, idx)

		log_losses = torch.mean(torch.stack(losses[-1]))
		log_ub_losses = torch.mean(torch.stack(ub_losses[-1]))
		log_r_losses = torch.mean(torch.stack(r_losses[-1]))
		log_r_ub_losses = torch.mean(torch.stack(r_ub_losses[-1]))

		log_others = np.mean(np.asarray(others), axis=0)
		log_others_ub = np.mean(np.asarray(others_ub), axis=0)
		log_r_others = np.mean(np.asarray(r_others), axis=0)
		log_r_others_ub = np.mean(np.asarray(r_others_ub), axis=0)

		print_log_dict(log_losses, log_ub_losses, log_r_losses, log_r_ub_losses, log_others, log_others_ub,
		               log_r_others, log_r_others_ub, loss_step_weights)


def run_from_file():
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + 1
	loss_step_weights = [1.0] * args.nr_steps

	# set up model
	model = NEM(batch_size=args.batch_size,
	            k=args.k,
	            input_size=(64, 64, 1),
	            hidden_size=args.inner_hidden_size,
	            device=device).to(device)

	# a model must be provided in order to run from file
	assert args.saved_model != None and args.saved_model != "", "Please provide a pre-trained model"
	saved_model_path = os.path.join(args.save_dir, args.saved_model)
	assert os.path.isfile(saved_model_path), "Path to model does not exist"
	model.load_state_dict(torch.load(saved_model_path))

	# TODO: record data using dictionary
	for epoch in range(1, args.max_epoch + 1):
		print("Starting epoch {}...".format(epoch))

		for b in range(Data.get_num_batches()):
			inputs = {
				attribute: Data(args.data_name, 'test', b, sequence_length=nr_iters, attribute=attribute)
				for attribute in attribute_list
			}

			# convert numpy bool array to tensor on GPU
			for k, v in inputs.items():
				inputs[k] = torch.from_numpy(v.data.astype(float)).float().to(device)

			# training phase
			features_corrupted = add_noise(inputs['features'], noise_type=args.noise_type)
			features = inputs['features']

			# TODO: convert into a log dict
			loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, \
			r_other_losses, r_other_ub_losses, train_model = nem_iterations(features_corrupted,
			                                                                features,
			                                                                model,
			                                                                collisions=inputs.get('collisions', None))

			print_log_dict(loss, ub_loss, r_loss, r_ub_loss, other_losses, other_ub_losses, r_other_losses,
			               r_other_ub_losses, loss_step_weights)


def run():
	for dir in [args.log_dir, args.save_dir]:
		utils.create_directory(dir)

	# only clear log_dir
	# utils.clear_directory(args.log_dir)

	# set up input data
	attribute_list = ('features', 'groups')
	nr_iters = args.nr_steps + 1

	# set up model
	train_model = NEM(batch_size=args.batch_size,
	                  k=args.k,
	                  input_size=(64, 64, 1),
	                  hidden_size=args.inner_hidden_size,
	                  device=device).to(device)

	if args.saved_model != None and args.saved_model != "":
		# load trained NEM model if exists
		saved_model_path = os.path.join(args.save_dir, args.saved_model)
		assert os.path.isfile(saved_model_path), "Path to model does not exist"
		train_model.load_state_dict(torch.load(saved_model_path))

	# training
	best_valid_loss = np.inf
	best_valid_epoch = 0
	loss_step_weights = [1.0] * args.nr_steps

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

			# convert numpy bool array to tensor on GPU
			for k, v in train_inputs.items():
				train_inputs[k] = torch.from_numpy(v.data.astype(float)).float().to(device)
			for k, v in valid_inputs.items():
				valid_inputs[k] = torch.from_numpy(v.data.astype(float)).float().to(device)

			# training phase
			features_corrupted = add_noise(train_inputs['features'], noise_type=args.noise_type)
			features = train_inputs['features']

			# TODO: convert into a log dict
			loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, \
			r_other_losses, r_other_ub_losses, train_model = nem_iterations(features_corrupted,
			                                                                features,
			                                                                train_model,
			                                                                collisions=train_inputs.get('collisions',
			                                                                                            None))

			# validation phase
			features_corrupted_valid = add_noise(valid_inputs['features'], noise_type=args.noise_type)
			features_valid = valid_inputs['features']

			loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, \
			r_other_losses, r_other_ub_losses, valid_model = nem_iterations(features_corrupted_valid,
			                                                                features_valid,
			                                                                train_model,
			                                                                collisions=valid_inputs.get('collisions',
			                                                                                            None))

			print_log_dict(loss, ub_loss, r_loss, r_ub_loss, other_losses, other_ub_losses, r_other_losses,
			               r_other_ub_losses, loss_step_weights)

			if loss < best_valid_loss:
				best_valid_loss = loss
				best_valid_epoch = epoch
				print("Best validation loss improved to %.03f" % best_valid_loss)
				print("Best valid epoch [{}/{}]".format(best_valid_epoch, args.max_epoch + 1))
				torch.save(train_model.state_dict(), os.path.abspath(os.path.join(args.save_dir, 'best.pth')))
				print("===Saved to:", args.save_dir)

			if epoch % args.log_per_iter == 0:
				print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch, args.max_epoch, b,
				                                                          Data.get_num_batches(), loss))
				torch.save(train_model.state_dict(),
				           os.path.abspath(os.path.join(args.save_dir, 'epoch_{}_batch_{}.pth'.format(epoch, b))))

			if np.isnan(loss.detach()):
				print("Early Stopping because validation loss is nan")
				break

	# # save on interrupt
	# print("Training interrupted. Saving model epoch_{}_batch_{}...".format(epoch, b))
	# torch.save(train_model.state_dict(), os.path.abspath(os.path.join(log_dir, 'E_epoch_{}_batch_{'
	#                                                                            '}.pth'.format(epoch, b))))


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
	parser.add_argument('--log_per_iter', type=int, default=1)
	parser.add_argument('--step_log_per_iter', type=int, default=10)
	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--data_batch_size', type=int, default=10)
	parser.add_argument('--inner_hidden_size', type=int, default=250)
	parser.add_argument('--saved_model', type=str, default='')
	parser.add_argument('--rollout_steps', type=int, default=10)
	parser.add_argument('--usage', '-u', choices=['train', 'eval', 'rollout'], required=True)

	### for testing purpose
	parser.add_argument('--cpu', default=False, action='store_true')

	args = parser.parse_args()
	print("=== Arguments ===")
	print(args)
	print()

	if args.cpu:
		use_gpu = False
	else:
		use_gpu = torch.cuda.is_available()
	device = torch.device('cuda' if use_gpu else 'cpu')

	if args.usage == 'train':
		run()
	elif args.usage == 'eval':
		run_from_file()
	elif args.usage == 'rollout':
		rollout_from_file()
	else:
		raise ValueError
