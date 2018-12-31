import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ACTIVATION_FUNCTIONS


class ReshapeModule(nn.RNN):
	def __init__(self, cell, shape, apply_to):
		self.cell = nn.RNN(cell.input_size, cell.hidden_size)
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
			return self.cell(x, state)

		elif self.apply_to == "output":
			x_out, next_state = self.cell(x, state)

			if self.shape == -1:
				x_out = x_out.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				x_out = x_out.view(reshape_size)

			return x_out, next_state

		elif self.apply_to == "state":
			x_out, next_state = self.cell(x, state)

			if self.shape == -1:
				next_state = next_state.view(batch_size, -1)
			else:
				reshape_size = (batch_size,) + self.shape
				next_state = next_state.view(reshape_size)

			return x_out, next_state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class ActivationFuncModule(nn.RNN):
	def __init__(self, cell, activation="linear", apply_to="output"):
		self.cell = nn.RNN(cell.input_size, cell.hidden_size)
		self.activation = ACTIVATION_FUNCTIONS[activation]
		self.apply_to = apply_to

	def __call__(self, x, state):
		if self.apply_to == "input":
			x = self.activation(x)
			return self.cell(x, state)

		elif self.apply_to == "output":
			x_out, next_state = self.cell(x, state)
			x_out = self.activation(x_out)
			return x_out, next_state

		elif self.apply_to == "state":
			x_out, next_state = self.cell(x, state)
			next_state = self.activation(next_state)
			return x_out, next_state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class LayerNormModule(nn.RNN):
	def __init__(self, cell, apply_to="output"):
		self.cell = nn.RNN(cell.input_size, cell.hidden_size)
		self.apply_to = apply_to

	def __call__(self, x, state):
		if self.apply_to == "input":
			x = F.layer_norm(x)
			return self.cell(x, state)

		elif self.apply_to == "output":
			x_out, next_state = self.cell(x, state)
			x_out = F.layer_norm(x_out)
			return x_out, next_state

		elif self.apply_to == "state":
			x_out, next_state = self.cell(x, state)
			next_state = F.layer_norm(next_state)
			return x_out, next_state

		else:
			raise ValueError("Unknown apply_to: {}".format(self.apply_to))


class EncoderLayer(nn.Module):
	def __init__(self, input_size):
		super(EncoderLayer, self).__init__()
		self.model = nn.Sequential(

			)
		self.conv1 = nn.Conv2d(input_size[-1], 16, kernel_size=4, stride=2)
		self.conv2 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(1, 64, kernel_size=4, stride=2)
		self.fc1 = nn.Linear()

	def forward(self, x, state):
		raise NotImplementedError


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


class InnerCAE(nn.Module):
	def __init__(self):
		super(InnerCAE, self).__init__()
		self.encoder = EncoderLayer()
		self.recurrent = RecurrentLayer()
		self.decoder = DecoderLayer()

	def forward(self, x):
		raise NotImplementedError


class NEMCell(nn.RNN):
	def __init__(self, size, K):
		self.encoder = nn.Sequential
		self.core = 
		self.context = 
		self.attention = 
		self.actions = 
		self.size = 

		assert K > 1
		self.K = K

	def forward(self, x):
		raise NotImplementedError