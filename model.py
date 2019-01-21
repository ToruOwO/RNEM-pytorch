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


class ReshapeWrapper(nn.Module):
	"""
	Reshape input Tensor to a given shape.

	:param: shape - Torch.size
	:param: apply_to - whether apply to x or state in __call__
	"""
	def __init__(self, shape, apply_to):
		super(ReshapeWrapper, self).__init__()
		self.shape = shape
		self.apply_to = apply_to

	def forward(self, x, state):
		batch_size = x.size()[0]

		if self.apply_to == "x":
			if self.shape == -1:
				x = x.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				x = x.view(reshape_size)
			return x, state

		elif self.apply_to == "state":
			if self.shape == -1:
				state = state.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				state = state.view(reshape_size)
			return x, state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class ActivationFunctionWrapper(nn.Module):
	def __init__(self, activation="linear", apply_to="x"):
		super(ActivationFunctionWrapper, self).__init__()
		self.activation = ACTIVATION_FUNCTIONS[activation]
		self.apply_to = apply_to

	def forward(self, x, state):
		if self.apply_to == "x":
			x = self.activation(x)
			return x, state

		elif self.apply_to == "state":
			state = self.activation(state)
			return x, state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class LayerNormWrapper(nn.Module):
	def __init__(self, apply_to="x"):
		super(LayerNormWrapper, self).__init__()
		self.apply_to = apply_to

	def forward(self, x, state):
		if self.apply_to == "x":
			x = F.layer_norm(x)
			return x, state

		elif self.apply_to == "state":
			state = F.layer_norm(state)
			return x, state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class InputWrapper(nn.Module):
	def __init__(self, input_size, output_size=None, fc_output_size=None):
		super(InputWrapper, self).__init__()

		if fc_output_size is None:
			self.main_layer = nn.Conv2d(input_size[-1], output_size, kernel_size=4, stride=2)
		else:
			self.main_layer = nn.Linear(input_size, fc_output_size)

		self.ln = LayerNormWrapper(apply_to="x")
		self.act = ActivationFunctionWrapper("elu", apply_to="x")

	def forward(self, x, state):
		# apply main layer to only input (x)
		x = self.main_layer(x)

		# apply layer norm
		x, state = self.ln(x, state)

		# apply activation function
		x, state = self.act(x, state)

		return x, state


class OutputWrapper(nn.Module):
	def __init__(self, input_size, output_size=None, fc_output_size=None, activation="relu", layer_norm=True):
		super(OutputWrapper, self).__init__()

		self.fc_output_size = fc_output_size

		if fc_output_size is None:
			self.main_layer = nn.Conv2d(input_size[-1], output_size, kernel_size=4, stride=2)
		else:
			self.main_layer = nn.Linear(input_size, fc_output_size)

		if layer_norm is True:
			self.ln = LayerNormWrapper(apply_to="x")

		self.act = ActivationFunctionWrapper(activation, apply_to="x")

	def forward(self, x, state):
		# apply main layer
		if self.fc_output_size is None:
			# resize image before Conv2D
			resized = F.interpolate(x, (2*x.size()[1], 2*x.size()[1]), mode="bilinear")
			projected = self.main_layer(resized)
		else:
			projected = self.main_layer(x)

		# apply layer norm
		projected, state = self.ln(projected, state)

		# apply activation function
		projected, state = self.act(projected, state)

		return projected, state


class R_NEM(nn.Module):
	def __init__(self, K, size=250):
		super(R_NEM, self).__init__()

		self.size = size

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
		bk = x.size()[0] # batch size * K
		m = x.size()[1] # np.prod(input_size.as_list())
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

		# encode theta
		state1 = self.encoder(state)

		# reshape theta to be used for context
		h1 = state1.size()[1]
		state1r = state1.view(b, k, h1)   # (b, k, h1)

		# reshape theta to be used for focus
		state1rr = state1r.view(b, k, 1, h1)   # (b, k, 1, h1)

		# create focus
		fs = state1rr.repeat(1, 1, k-1, 1)   # (b, k, k-1, h1)

		# create context
		state1rl = torch.unbind(state1r, dim=1)   # list of length k of (b, h1)

		if k > 1:
			csu = []
			for i in range(k):
				selector = [j for j in range(k) if j != i]
				c = list(np.take(state1rl, selector)) # list of length k-1 of (b, h1)
				c = torch.stack(c, dim=1)
				csu.append(c)

			cs = torch.stack(csu, dim=1)
		else:
			cs = torch.zeros(b, k, k-1, h1)

		# reshape focus and context
		fsr, csr = fs.view(b*k*(k-1), h1), cs.view(b*k*(k-1), h1)   # (b*k*(k-1), h1)

		# concatenate focus and context
		concat = torch.cat([fsr, csr], dim=1)   # (b*k*(k-1), 2*h1)

		# core
		core_out = self.core(concat)

		# context; obtained from core_out
		context_out = self.context(core_out)

		h2 = 250
		contextr = context_out.view(b*k, k-1, h2)   # (b*k, k-1, h2)

		# attention coefficients; obtained from core_out
		attention_out = self.attention(core_out)

		# produce effect as sum(context_out * attention_out)
		attentionr = attention_out.view(b*k, k-1, 1)
		effect_sum = torch.sum(contextr * attentionr, dim=1)

		# calculate new state (where the input from encoder comes in)
		# concatenate (state1, effect_sum, inputs)
		total = torch.cat([state1, effect_sum, inputs], dim=1)   # (b*k, h+h2+m)

		# produce recurrent update
		out_fc = nn.Linear((b*k, h+h2+m), self.size) # (b*k, h)
		new_state = out_fc(total)

		return new_state, new_state


class EncoderLayer(nn.Module):
	def __init__(self):
		super(EncoderLayer, self).__init__()
		self.reshape1 = ReshapeWrapper((64, 64, 1), apply_to="x")
		self.conv1 = None
		self.conv2 = None
		self.conv3 = None
		self.reshape2 = None
		self.fc1 = None

	def forward(self, x, state):
		# reshape the input to (64, 64, 1)
		x, state = self.reshape1(x, state)

		# normal convolution
		self.conv1 = InputWrapper(x.size(), output_size=16)
		x, state = self.conv1(x, state)

		self.conv2 = InputWrapper(x.size(), output_size=32)
		x, state = self.conv2(x, state)

		self.conv3 = InputWrapper(x.size(), output_size=64)
		x, state = self.conv3(x, state)

		# flatten input
		self.reshape2 = ReshapeWrapper(-1, apply_to="x")
		x, state = self.reshape2(x, state)

		# linear layer
		self.fc1 = InputWrapper(x.size(), fc_output_size=512)
		x, state = self.fc1(x, state)

		return x, state


class DecoderLayer(nn.Module):
	def __init__(self, input_size):
		super(DecoderLayer, self).__init__()
		self.fc1 = OutputWrapper(input_size, fc_output_size=512)
		self.fc2 = None
		self.reshape1 = None
		self.r_conv1 = None
		self.r_conv2 = None
		self.r_conv3 = None
		self.reshape2 = None

	def forward(self, x, state):
		x, state = self.fc1(x, state)

		self.fc2 = OutputWrapper(x.size(), fc_output_size=8*8*64)
		x, state = self.fc2(x, state)

		self.reshape1 = ReshapeWrapper((8, 8, 64), apply_to="x")
		x, state = self.reshape1(x, state)

		self.r_conv1 = OutputWrapper(x.size(), output_size=32)
		x, state = self.r_conv1(x, state)

		self.r_conv2 = OutputWrapper(x.size(), output_size=16)
		x, state = self.r_conv2(x, state)

		self.r_conv3 = OutputWrapper(x.size(), output_size=1, activation="sigmoid", layer_norm=False)
		x, state = self.r_conv3(x, state)

		self.reshape2 = ReshapeWrapper(-1, apply_to="x")
		x, state = self.reshape2(x, state)

		return x, state


class RecurrentLayer(nn.Module):
	def __init__(self, hidden_size, K):
		super(RecurrentLayer, self).__init__()
		self.r_nem = R_NEM(K)
		self.layer_norm = None
		self.act1 = None
		self.act2 = None

	def forward(self, x, state):
		x, state = self.r_nem(x, state)

		self.layer_norm = LayerNormWrapper(apply_to="x")
		x, state = self.layer_norm(x, state)

		self.act1 = ActivationFunctionWrapper("sigmoid", apply_to="state")
		x, state = self.act1(x, state)

		self.act2 = ActivationFunctionWrapper("sigmoid", apply_to="x")
		x, state = self.act2(x, state)

		return x, state


class InnerRNN(nn.Module):
	def __init__(self, input_size, hidden_size, K):
		super(InnerRNN, self).__init__()

		self.encoder = EncoderLayer()
		self.recurrent = RecurrentLayer(hidden_size, K)
		self.decoder = DecoderLayer(input_size)

	def forward(self, x, state):
		x, state = self.encoder(x, state)
		x, state = self.recurrent(x, state)
		x, state = self.decoder(x, state)

		return x, state