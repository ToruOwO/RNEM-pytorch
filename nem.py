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

	def init_state(self, batch_size, K, dtype=torch.float):
		h = torch.zeros(batch_size*K, *self.hidden_size, dtype=dtype)

		pred = torch.zeros(batch_size, K, *self.input_size, dtype=dtype)

		# initialize with Gaussian distribution
		gamma_shape = [batch_size, K] + list(self.gamma_size)
		gamma = np.absolute(np.random.normal(size=gamma_shape))
		gamma = torch.from_numpy(gamma)
		gamma = torch.sum(gamma, dim=1, keepdim=True)

		# init with all 1 if K = 1
		if K == 1:
			gamma = torch.ones_like(gamma)

		return h, pred, gamma

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


def nem_iterations(input_data, target_data, collisions=None, learning_rate=0.001, nr_steps=30, num_epochs=500, k=5, is_training=True):
	# get input dimensions
	input_shape = input_data.size()
	assert input_shape[0] == 6, "Requires 6D input (T, B, K, W, H, C) but {}".format(input_shape[0])
	W, H, C = (x for x in input_shape[-3:])

	# set initial distribution (Bernoulli) of pixels
	inner_model = InnerConvAE(K=k)
	nem_model = NEM(inner_model, input_size=(W, H, C), inner_model.hidden_size, inner_model.input_size)

	# compute Bernoulli prior
	prior = compute_bernoulli_prior()

	# use binomial cross entropy as intra loss
	intra_criterion = nn.BCELoss()

	# use KL divergence as inter loss
	inter_criterion = nn.KLDivLoss()

	# use Adam optimizer
	optimizer = optim.Adam(nem_model.parameters(), lr=learning_rate)

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

	for epoch in range(num_epochs):
		if is_training:
			model.train()

		losses = 0.0

		for t in range(nr_steps):
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
	            print ('Epoch [{}/{}], Loss: {:.4f}' 
	                   .format(epoch+1, num_epochs, loss.item()))

		losses /= len(input_data)
		print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
              (phase, epoch, args.n_epoch, losses, best_valid_loss))

	thetas, preds, gammas = zip(*outputs)
	thetas = torch.stack((thetas,))
	preds = torch.stack((preds,))
	gammas = torch.stack((gammas,))

	return loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
		 r_other_losses, r_other_ub_losses


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

	loss, ub_loss, r_loss, r_ub_loss, thetas, preds, gammas, other_losses, other_ub_losses,\
	r_other_losses, r_other_ub_losses = nem_iterations(features_corrupted, 
												    	features, 
												    	collision=train_inputs.get('collisions', None),
												    	learning_rate=lr, 
												    	nr_steps=nr_steps, 
												    	num_epochs=max_epoch)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_name' type=str, default='balls3curtain64')
	parser.add_argument('--log_dir', type=str, default='./debug')
	parser.add_argument('--nr_steps', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=500)

    config = parser.parse_args()
    print(config)
	main(config.data_name, config.log_dir, config.nr_steps, config.batch_size, config.lr, config.max_epoch)