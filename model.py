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
	def __init__(self, input_size, hidden_size, output_size=None, fc_output_size=None):
		super(InputWrapper, self).__init__(input_size, hidden_size)

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


class OutputWrapper(nn.RNN):
	def __init__(self, input_size, hidden_size, output_size=None, fc_output_size=None, activation="relu", layer_norm=True):
		super(OutputWrapper, self).__init__(input_size, hidden_size)

		modules = []

		if fc_output_size is None:
			modules.append(nn.Conv2d(input_size[-1], output_size, kernel_size=4, stride=2))
		else:
			modules.append(nn.Linear(input_size, fc_output_size))

		if layer_norm is True:
			modules.append(LayerNormWrapper(input_size, hidden_size, apply_to="output"))

		modules.append(ActivationFunctionWrapper(input_size, hidden_size, activation=activation, apply_to="output"))

		self.module = nn.Sequential(*modules)

	def forward(self, x):
		return self.model(x)


class R_NEM(nn.RNN):
	def __init__(self, K, size=250):
		super(R_NEM, self).__init__(size, size)

		self.encoder = nn.Sequential(
			nn.Linear(size, 250),
			nn.ReLU(),
			nn.LayerNorm(250)
		)
		self.core = nn.Sequential(
			nn.Linear(250, 250),
			nn.ReLU(),
			nn.LayerNorm(250)
		)
		self.context = nn.Sequential(
			nn.Linear(250, 250),
			nn.ReLU(),
			nn.LayerNorm(250)
		)
		self.attention = nn.Sequential(
			nn.Linear(250, 100),
			nn.Tanh(),
			nn.LayerNorm(250),
			nn.Linear(100, 1),
			nn.Sigmoid()
		)

		assert K > 1
		self.K = K

	def get_shapes(self, x):
		bk = x.size()[0]
		m = x.size()[1]
		return bk // self.K, self.K, m

	def forward(self, x, state):
		"""
		input: [B X K, M]
		state: [B x K, H]

		b: batch_size
		k: num_groups
		m: input_size
		h: hidden_size
		h1: size of the encoding of focus and context
		h2: size of effect
		o: size of output

		# 0. Encode with RNN: x is [B*K, M], h is [B*K, H] --> both are [B*K, H]
		# 1. Reshape both to [B, K, H]
		# 2. For each of the k in K copies, extract the K-1 states that are not that k
		# 3. Now you have two tensors of size [B x K x K-1, H]
		#     The first: "focus object": K-1 copies of the state of "k", the focus object
		#     The second: "context objects": K-1 (all unique) states of the context objects
		# 4. Concatenate results of 3
		# 5. Core: Process result of 4 in a feedforward network --> [B x K, H'']
		# 6. Reshape to [B x K, K-1, H''] to isolate the K-1 dimension (because we did for K-1 pairs)
		# 7. Sum in the K-1 dimension --> [B x K, H'']
		#   7.5 weighted by attention
		# 8. Decoder: Concatenate result of 7, the original theta, and the x and process into new state --> [B x K, H]
		# 9. Actions: Optionally embed actions into some representation

		"""
		b, k, m = self.get_shapes(x)

		x, state = self.rnn(x, state)


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
		self.conv1 = InputWrapper(x.input_size, x.hidden_size, output_size=16)
		x = self.conv1(x)

		self.conv2 = InputWrapper(x.input_size, x.hidden_size, output_size=32)
		x = self.conv2(x)

		self.conv3 = InputWrapper(x.input_size, x.hidden_size, output_size=64)
		x = self.conv3(x)

		# flatten input
		self.reshape2 = ReshapeWrapper(x.input_size, state.hidden_size, shape=-1, apply_to="input")
		x = self.reshape2(x)

		# linear layer
		self.fc1 = InputWrapper(x.input_size, x.hidden_size, fc_output_size=512)
		x = self.fc1(x)

		return x


class DecoderLayer(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(DecoderLayer, self).__init__()
		self.fc1 = OutputWrapper(input_size, hidden_size, fc_output_size=512)
		self.fc2 = None
		self.reshape1 = None
		self.r_conv1 = None
		self.r_conv2 = None
		self.r_conv3 = None
		self.reshape2 = None

	def forward(self, x):
		x = self.fc1(x)

		self.fc2 = OutputWrapper(x.input_size, x.hidden_size, fc_output_size=8*8*64)
		x = self.fc2(x)

		self.reshape1 = ReshapeWrapper(x.input_size, x.hidden_size, shape=(8, 8, 64), apply_to="output")
		x = self.reshape1(x)

		self.r_conv1 = OutputWrapper(x.input_size, x.hidden_size, output_size=32)
		x = self.r_conv1(x)

		self.r_conv2 = OutputWrapper(x.input_size, x.hidden_size, output_size=16)
		x = self.r_conv2(x)

		self.r_conv3 = OutputWrapper(x.input_size, x.hidden_size, output_size=1, activation="sigmoid", layer_norm=False)
		x = self.r_conv3(x)

		self.reshape2 = ReshapeWrapper(x.input_size, x.hidden_size, shape=-1, apply_to="output")
		x = self.reshape2(x)

		return x


class RecurrentLayer(nn.Module):
	def __init__(self, K):
		super(RecurrentLayer, self).__init__()
		self.r_nem = R_NEM(K)
		self.layer_norm = None
		self.act1 = None
		self.act2 = None

	def forward(self, x):
		x = self.r_nem(x)

		self.layer_norm = LayerNormWrapper(x.input_size, x.output_size, apply_to="output")
		x = self.layer_norm(x)

		self.act1 = ActivationFunctionWrapper(x.input_size, x.output_size, activation="sigmoid", apply_to="state")
		x = self.act1(x)

		self.act2 = ActivationFunctionWrapper(x.input_size, x.output_size, activation="sigmoid", apply_to="output")
		x = self.act2(x)

		return x


class InnerConvAE(nn.RNN):
	def __init__(self, input_size, hidden_size, K):
		super(InnerCAE, self).__init__(input_size, hidden_size)
		self.encoder = EncoderLayer(input_size, hidden_size)
		self.recurrent = RecurrentLayer(K)
		self.decoder = DecoderLayer(input_size, hidden_size)

	def forward(self, x):
		raise NotImplementedError