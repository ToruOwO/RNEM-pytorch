import argparse
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

from model import InnerConvAE

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NEM(nn.RNN):
	def __init__(self, input_size, hidden_size, output_size):
		gamma_size = input_size[:-1] + (1,)
		rnn_input_size = (hidden_size, input_size, gamma_size)
		rnn_hidden_size = (output_size, input_size, gamma_size)

		super(ReshapeWrapper, self).__init__(rnn_input_size, rnn_hidden_size)

		self.input_size = input_size
		self.gamma_size = gamma_size

	@staticmethod
	def delta_predictions(predicitons, data):
		"""
		Compute the derivative of the prediction wrt. to the loss.
		For binary and real with just μ this reduces to (predictions - data).

		:param predictions: (B, K, W, H, C)
		Note: This is a list to later support getting both μ and σ.
		:param data: (B, 1, W, H, C)

		:return: deltas (B, K, W, H, C)
		"""
		return data - predictions

	@staticmethod
	def mask_rnn_inputs(rnn_inputs, gamma):
		"""
		Mask the deltas (inputs to RNN) by gamma.
        :param rnn_inputs: (B, K, W, H, C)
            Note: This is a list to later support multiple inputs
        :param gamma: (B, K, W, H, 1)

        :return: masked deltas (B, K, W, H, C)
		"""
		with torch.no_grad():
			return rnn_inputs * gamma

	def run_inner_rnn(self, masked_deltas, h_old):
		d_size = masked_deltas.size()
		batch_size = d_size[0]
		K = d_size[1]
		M = torch.tensor(self.input_size).prod()
		reshaped_masked_deltas = masked_deltas.view(torch.stack([batch_size * K, M]))

		preds, h_new = self.forward(reshaped_masked_deltas, h_old)

		return preds.view(d_size), h_new

	def compute_em_probabilities(self, predictions, data, epsilon=1e-6):
		"""
		Compute pixelwise loss of predictions (wrt. the data).

        :param predictions: (B, K, W, H, C)
        :param data: (B, 1, W, H, C)
        :return: local loss (B, K, W, H, 1)
		"""
		loss = data * predictions + (1 - data) * (1 - predictions)
		if epsilon > 0:
			loss += epsilon
		return loss

	def e_step(self, predictions, targets):
		probs = self.compute_em_probabilities(predictions, targets)

		# compute the new gamma (E-step)
		gamma = probs / probs.sum(dim=1, keepdim=True)

		return gamma

	def __call__(self, x, state):
		# unpack values
		input_data, target_data = x
		h_old, preds_old, gamma_old = state

		# compute differences between prediction and input
		deltas = self.delta_predictions(preds_old, input_data)

		# mask with gamma
		masked_deltas = self.mask_rnn_inputs(deltas, gamma_old)

		# compute new predictions
		preds, h_new = self.run_inner_rnn(masked_deltas, h_old)

		# compute the new gammas
		gamma = self.e_step(preds, target_data)

		# pack and return
		outputs = (h_new, preds, gamma)

		return outputs, outputs


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


def nem_iterations(input_data, target_data, learning_rate=0.001, num_epochs=500, k=5, is_training=True):
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

	best_valid_loss = np.inf
	for epoch in range(num_epochs):
		if is_training:
			model.train()

		losses = 0.

		for i, data in enumerate(input_data):

			# forward pass
			outputs = nem_model(data)
			total_loss = intra_criterion(outputs, labels) + inter_criterion(outputs, labels)
			losses += total_loss

			# backward pass and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# print log
			if (i+1) % 100 == 0:
	            print ('Epoch [{}/{}], Loss: {:.4f}' 
	                   .format(epoch+1, num_epochs, loss.item()))

		losses /= len(input_data)
		print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
              (phase, epoch, args.n_epoch, losses, best_valid_loss))


def main(data_name, log_dir, nr_steps, batch_size, lr, max_epoch, noise_type='bitflip'):
	if log_dir is None:
		utils.create_directory(log_dir)
		utils.clear_directory(log_dir)

	# set up input data
	attribute_list = ['features', 'groups']

	train_inputs = Data(data_name, 'training', attribute_list=attribute_list)
	valid_inputs = Data(data_name, 'validation', attribute_list=attribute_list)

	# build model
	model = InnerConvAE()

	# training
	features_corrupted = add_noise(train_inputs['features'], noise_type=noise_type)
	loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses, r_other_losses, \
    r_other_ub_losses = nem_iterations(features_corrupted, features, learning_rate=lr, num_epochs=max_epoch)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name' type=str, default='balls3curtain64')
	parser.add_argument('--log_dir', type=str, default='./debug')
	parser.add_argument('--nr_steps', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=500)

    config = parser.parse_args()
    print(config)
	main(config.data_name, config.log_dir, config.nr_steps, config.batch_size, config.lr, config.max_epoch)