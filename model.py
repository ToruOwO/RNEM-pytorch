import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# dict of activation functions
ACTIVATION_FUNCTIONS = {
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'relu': F.relu,
    'elu': F.elu,
    'linear': lambda x: x,
    'exp': lambda x: torch.exp(x),
    'softplus': F.softplus,
    'clip': lambda x: torch.clamp(x, min=-1., max=1.),
    'clip_low': lambda x: torch.clamp(x, min=-1., max=1e6)
}


class ReshapeWrapper(nn.RNN):
	def __init__(self, input_size, hidden_size, shape, apply_to):
		super(ReshapeWrapper, self).__init__(input_size, hidden_size)
		self.shape = shape
		self.apply_to = apply_to

	def __call__(self, x, state):
		batch_size = x.size()[0]
		if self.apply_to == "input":
			if self.shape == -1:
				x = x.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				x = x.view(reshape_size)
			return self.forward(x, state)

		elif self.apply_to == "output":
			x_out, next_state = self.forward(x, state)

			if self.shape == -1:
				x_out = x_out.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				x_out = x_out.view(reshape_size)

			return x_out, next_state

		elif self.apply_to == "state":
			x_out, next_state = self.forward(x, state)

			if self.shape == -1:
				next_state = next_state.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				next_state = next_state.view(reshape_size)

			return x_out, next_state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class ActivationFunctionWrapper(nn.RNN):
	def __init__(self, input_size, hidden_size, activation="linear", apply_to="output"):
		super(ActivationFunctionWrapper, self).__init__(input_size, hidden_size)
		self.activation = ACTIVATION_FUNCTIONS[activation]
		self.apply_to = apply_to

	def __call__(self, x, state):
		if self.apply_to == "input":
			x = self.activation(x)
			return self.forward(x, state)

		elif self.apply_to == "output":
			x_out, next_state = self.forward(x, state)
			x_out = self.activation(x_out)
			return x_out, next_state

		elif self.apply_to == "state":
			x_out, next_state = self.forward(x, state)
			next_state = self.activation(next_state)
			return x_out, next_state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class LayerNormWrapper(nn.RNN):
	def __init__(self, input_size, hidden_size, apply_to="output"):
		super(LayerNormModule, self).__init__(input_size, hidden_size)
		self.apply_to = apply_to

	def __call__(self, x, state):
		if self.apply_to == "input":
			x = F.layer_norm(x)
			return self.forward(x, state)

		elif self.apply_to == "output":
			x_out, next_state = self.forward(x, state)
			x_out = F.layer_norm(x_out)
			return x_out, next_state

		elif self.apply_to == "state":
			x_out, next_state = self.forward(x, state)
			next_state = F.layer_norm(next_state)
			return x_out, next_state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class InputWrapper(nn.RNN):
	def __init(self, input_size, hidden_size, output_size, fc_output_size=None):
		super(LayerNormModule, self).__init__(input_size, hidden_size)

		if fc_output_size is None:
			main_layer = nn.Conv2d(input_size[-1], output_size, kernel_size=4, stride=2)
		else:
			main_layer = nn.Linear(input_size, fc_output_size)

		self.model = nn.Sequential(
			ActivationFunctionWrapper(input_size, hidden_size, activation="elu", apply_to="input"),
			LayerNormWrapper(input_size, hidden_size, apply_to="input"),
			main_layer)

	def forward(self, x):
		return self.model(x)


class EncoderLayer(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderLayer, self).__init__()
		self.reshape1 = ReshapeWrapper(input_size, hidden_size, shape=(64, 64, 1), apply_to="input")
		self.conv1 = None
		self.conv2 = None
		self.conv3 = None
		self.reshape2 = None
		self.fc1 = None

	def forward(self, x):
		# reshape the input to (64, 64, 1)
		x = self.reshape1(x)

		# normal convolution
		self.conv1 = InputWrapper(x.input_size, x.hidden_size, 16)
		x = self.conv1(x)

		self.conv2 = InputWrapper(x.input_size, x.hidden_size, 32)
		x = self.conv2(x)

		self.conv3 = InputWrapper(x.input_size, x.hidden_size, 64)
		x = self.conv3(x)

		# flatten input
		self.reshape2 = ReshapeModule(x.input_size, state.hidden_size, shape=-1, apply_to="input")
		x = self.reshape2(x)

		# linear layer
		self.fc1 = InputWrapper(x.input_size, x.hidden_size, fc_output_size=512)
		x = self.fc1(x)

		return x


class DecoderLayer(nn.Module):
	def __init__(self):
		super(DecoderLayer, self).__init__()

	def forward(self, x):
		raise NotImplementedError


class RecurrentLayer(nn.Module):
	def __init__(self):
		super(RecurrentLayer, self).__init__()

	def forward(self, x):
		raise NotImplementedError


class InnerConvAE(nn.RNN):
	def __init__(self, input_cell):
		super(InnerCAE, self).__init__(input_cell.input_size, input_cell.hidden_size)
		self.encoder = EncoderLayer()
		self.recurrent = RecurrentLayer()
		self.decoder = DecoderLayer()

	def forward(self, x):
		raise NotImplementedError