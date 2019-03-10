# import cv2
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

data_names = {'balls4mass64', 'balls678mass64', 'balls3curtain64', 'atari', '3balls'}
data_path = "./data"


class Data(Dataset):
	"""
	Read in h5 data, which is consisted of 50000 sequences of 51 frames.
	
	For bouncing balls dataset, use "features" and "groups" attribute for
	training and validation.

	Attributes:
		data_name (str): name of dateset (e.g. balls4mass64)
		phase(str): "training" or "validation"
		sequence_length(int): number of frames to use
		attribute(tuple): attribute used
	"""

	def __init__(self, data_name, phase, batch_size, sequence_length=51, attributes=('features', 'groups')):
		if data_name not in data_names:
			print("Dataset does not exist")

		self.data_name = data_name
		self.phase = phase
		self.batch_size = batch_size
		self.sequence_length = sequence_length

		self._load_dataset(attributes)

	def _load_dataset(self, attributes):
		"""
		Return a dictionary of data

		keys (str): attribute
		values (array): numerical data

		Shape of data: (T, B, K, W, H, C)
		T - sequence length
		B - batch size
		K - group size (default=1)
		W - width
		H - height
		C - number of channels
		"""
		print("Loading data from file ...")
		file_path = os.path.join(data_path, self.data_name + '.h5')
		self.hf = h5py.File(file_path, 'r')

		# shape of data is (T, B, K, W, H, C)
		self.data = {
			attr: self.hf[self.phase][attr] for attr in attributes
		}

		print("Shape of feature data:", self.data["features"].shape)

		self.limit = self.data["features"].shape[1]

	def __getitem__(self, idx):
		d = {dn: torch.from_numpy(
			ds[:self.sequence_length, self.batch_size*idx:self.batch_size*(idx+1)][:, :, None].astype(np.float32))
			for dn, ds in self.data.items()}
		return d

	def __len__(self):
		# number of batches
		return self.limit // self.batch_size - 1


def collate(batch):
	return batch