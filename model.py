import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape(shape, x):
	batch_size = x.size()[0]

	if shape == -1:
		return x.view(batch_size, -1)
	else:
		reshape_size = (batch_size,) + shape
		return x.view(reshape_size)


class InputWrapper(nn.Module):
	def __init__(self, input_size, output_size=None, fc_output_size=None):
		super(InputWrapper, self).__init__()

		if fc_output_size is None:
			self.main_layer = nn.Conv2d(input_size[-1], output_size, kernel_size=4, stride=2, padding=1)
		else:
			self.main_layer = nn.Linear(input_size[-1], fc_output_size)

		self.act = nn.ELU()

	def forward(self, x, state):
		# apply main layer to only input (x)

		# since input size for Conv2D layer is (B, C, H, W),
		# reshape x from (B, W, H, C) to (B, C, W, H)
		reshape_dim = tuple([0] + [i for i in range(len(x.size()) - 1, 0, -1)])

		x = self.main_layer(x.permute(reshape_dim))

		# since output size for Conv2D layer is (B, C, H, W),
		# reshape output back to (B, W, H, C)
		x = x.permute(reshape_dim)

		del reshape_dim

		# apply layer norm
		x = F.layer_norm(x, x.size()[1:])

		# apply activation function
		x = self.act(x)

		return x, state


class OutputWrapper(nn.Module):
	def __init__(self, input_size, output_size=None, fc_output_size=None, activation="relu", layer_norm=True):
		super(OutputWrapper, self).__init__()

		self.fc_output_size = fc_output_size

		if fc_output_size is None:
			self.main_layer = nn.Conv2d(input_size[-1], output_size, kernel_size=4, stride=2, padding=1)
		else:
			self.main_layer = nn.Linear(input_size[-1], fc_output_size)

		self.ln = layer_norm

		if activation == "relu":
			self.act = nn.ReLU()
		elif activation == "sigmoid":
			self.act = nn.Sigmoid()

	def forward(self, x, state):
		# apply main layer
		if self.fc_output_size is None:
			# Conv2d
			# since input size for Conv2D layer is (B, C, H, W),
			# reshape "resized" from (B, W, H, C) to (B, C, H, W)
			resized = x.permute(0, 3, 2, 1)
			resized = F.interpolate(resized, (4 * x.size()[1], 4 * x.size()[2]), mode="nearest")

			projected = self.main_layer(resized)

			del resized

			# since output size for Conv2D layer is (B, C, H, W),
			# reshape "projected" back to (B, W, H, C)
			projected = projected.permute(0, 3, 2, 1)
		else:
			# Linear
			projected = self.main_layer(x)

		del x

		if self.ln:
			# apply layer norm
			projected = F.layer_norm(projected, projected.size()[1:])

		# apply activation function
		projected = self.act(projected)

		return projected, state


class R_NEM(nn.Module):
	def __init__(self, k, fc_size=250, last_fc_size=100, device='cpu'):
		super(R_NEM, self).__init__()

		self.device = device

		self.fc_size = fc_size
		self.last_fc_size = last_fc_size

		self.encoder = nn.Sequential(
			nn.Linear(fc_size, fc_size),
			nn.LayerNorm(fc_size),
			nn.ReLU()
		).to(device)

		self.core = nn.Sequential(
			nn.Linear(2 * fc_size, fc_size),
			nn.LayerNorm(fc_size),
			nn.ReLU()
		).to(device)

		self.context = nn.Sequential(
			nn.Linear(fc_size, fc_size),
			nn.LayerNorm(fc_size),
			nn.ReLU()
		).to(device)

		self.attention = nn.Sequential(
			nn.Linear(fc_size, last_fc_size),
			nn.LayerNorm(last_fc_size),
			nn.Tanh(),
			nn.Linear(last_fc_size, 1),
			nn.Sigmoid()
		).to(device)

		self.out_fc = nn.Linear(fc_size + fc_size + 512, fc_size).to(device)

		assert k > 1
		self.k = k

	def get_shapes(self, x):
		bk = x.size()[0]  # batch size * K
		m = x.size()[1]  # np.prod(input_size.as_list())
		return bk // self.k, self.k, m

	def forward(self, x, state):
		"""
		x: [B X K, M]
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
		state1 = self.encoder(state)  # (b*k, h1)

		# reshape theta to be used for context
		h1 = state1.size()[1]
		state1r = state1.view(b, k, h1)  # (b, k, h1)

		# reshape theta to be used for focus
		state1rr = state1r.view(b, k, 1, h1)  # (b, k, 1, h1)

		# create focus
		fs = state1rr.repeat(1, 1, k - 1, 1)  # (b, k, k-1, h1)

		# create context
		state1rl = torch.unbind(state1r, dim=1)  # tuple of length k of (b, h1)

		if k > 1:
			csu = []
			for i in range(k):
				# selector = [j for j in range(k) if j != i]
				# c = list(torch.take(torch.Tensor(state1rl), selector))  # list of length k-1 of (b, h1)
				c = [state1rl[j] for j in range(k) if j != i]
				c = torch.stack(c, dim=1)
				csu.append(c)

			cs = torch.stack(csu, dim=1)
		else:
			cs = torch.zeros(b, k, k - 1, h1)

		# reshape focus and context
		fsr, csr = fs.view(b * k * (k - 1), h1), cs.view(b * k * (k - 1), h1)  # (b*k*(k-1), h1)

		# concatenate focus and context
		concat = torch.cat([fsr, csr], dim=1)  # (b*k*(k-1), 2*h1)

		# core
		core_out = self.core(concat)  # (b*k*(k-1), h1)

		# context; obtained from core_out
		context_out = self.context(core_out)  # (b*k*(k-1), h2)

		h2 = self.fc_size
		contextr = context_out.view(b * k, k - 1, h2)  # (b*k, k-1, h2)

		# attention coefficients; obtained from core_out
		attention_out = self.attention(core_out)  # (b*k*(k-1), 1)

		# produce effect as sum(context_out * attention_out)
		attentionr = attention_out.view(b * k, k - 1, 1)
		effect_sum = torch.sum(contextr * attentionr, dim=1)  # (b*k, h2)

		# calculate new state (where the input from encoder comes in)
		# concatenate (state1, effect_sum, inputs)
		total = torch.cat([state1, effect_sum, x], dim=1)  # (b*k, h1+h2+m)

		# produce recurrent update
		new_state = self.out_fc(total)  # (b*k, h)

		# delete used variables to save memory space
		del state1, state1r, state1rr, fs, cs, fsr, csr, concat, core_out, context_out, contextr, attention_out, \
			attentionr, effect_sum, total

		return new_state, new_state


class EncoderLayer(nn.Module):
	def __init__(self, batch_size, k, input_size, device='cpu'):
		super(EncoderLayer, self).__init__()
		self.device = device
		self.input_size = input_size

		W, H, C = input_size

		self.conv1 = InputWrapper((batch_size * k, W, H, 1), output_size=16).to(device)
		self.conv2 = InputWrapper((batch_size * k, W // 2, H // 2, 16), output_size=32).to(device)
		self.conv3 = InputWrapper((batch_size * k, W // 4, H // 4, 32), output_size=64).to(device)
		self.fc1 = InputWrapper((batch_size * k, W * H), fc_output_size=512).to(device)

	def forward(self, x, state):
		# reshape the input to (64, 64, 1)
		x = reshape(self.input_size, x)

		# normal convolution
		x, state = self.conv1(x, state)
		x, state = self.conv2(x, state)
		x, state = self.conv3(x, state)

		# flatten input
		x = reshape(-1, x)

		# linear layer
		x, state = self.fc1(x, state)

		return x, state


class DecoderLayer(nn.Module):
	def __init__(self, batch_size, k, input_size, hidden_size, device='cpu'):
		super(DecoderLayer, self).__init__()
		W, H, C = input_size

		self.fc1 = OutputWrapper((batch_size * k, hidden_size), fc_output_size=512).to(device)
		self.fc2 = OutputWrapper((batch_size * k, 512), fc_output_size=W * H).to(device)
		self.r_conv1 = OutputWrapper((batch_size * k, 8, 8, 64), output_size=32).to(device)
		self.r_conv2 = OutputWrapper((batch_size * k, 16, 16, 32), output_size=16).to(device)
		self.r_conv3 = OutputWrapper((batch_size * k, 32, 32, 16), output_size=1, activation="sigmoid",
		                             layer_norm=False).to(device)

	def forward(self, x, state):
		x, state = self.fc1(x, state)
		x, state = self.fc2(x, state)
		x = reshape((8, 8, 64), x)
		x, state = self.r_conv1(x, state)
		x, state = self.r_conv2(x, state)
		x, state = self.r_conv3(x, state)
		x = reshape(-1, x)

		return x, state


class RecurrentLayer(nn.Module):
	def __init__(self, k, hidden_size, device='cpu'):
		super(RecurrentLayer, self).__init__()
		self.r_nem = R_NEM(k, fc_size=hidden_size, device=device)
		self.act = nn.Sigmoid()

	def forward(self, x, state):
		x, state = self.r_nem(x, state)
		x = F.layer_norm(x, x.size()[1:])
		x = self.act(x)
		state = self.act(state)
		return x, state


class InnerRNN(nn.Module):
	def __init__(self, batch_size, k, input_size, hidden_size, device='cpu'):
		super(InnerRNN, self).__init__()

		self.batch_size = batch_size
		self.k = k
		self.hidden_size = hidden_size
		self.device = device

		self.encoder = EncoderLayer(batch_size, k, input_size, device=device)
		self.recurrent = RecurrentLayer(k, hidden_size, device=device)
		self.decoder = DecoderLayer(batch_size, k, input_size, hidden_size, device=device)

	def init_hidden(self):
		# variable of size [num_layers*num_directions, b_sz, hidden_sz]
		if self.device.type == 'cpu':
			return torch.autograd.Variable(torch.zeros(self.batch_size * self.k, self.hidden_size))
		else:
			return torch.autograd.Variable(torch.zeros(self.batch_size * self.k, self.hidden_size)).cuda()

	def forward(self, x, state):
		x, state = self.encoder(x, state)
		x, state = self.recurrent(x, state)
		x_out, state_out = self.decoder(x, state)

		# delete used variables to save memory space
		del x, state
		return x_out, state_out
