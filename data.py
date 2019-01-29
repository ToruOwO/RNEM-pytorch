# import cv2
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

data_names = {'balls4mass64', 'balls678mass64', 'balls3curtain64', 'atari'}
data_path = "./data"
train_size = None
valid_size = None
test_size = None
batch_size = 64

class Data(Dataset):
	"""
	Read in h5 data, which is consisted of 50000 sequences of 51 frames.
	
	For bouncing balls dataset, use "features" and "groups" attribute for
	training and validation.

	Attributes:
		data_name (str): name of dateset (e.g. balls4mass64)
		phase(str): "training" or "validation"
		sequence_length(int): number of frames to use
		attribute(str): attribute used
	"""
	def __init__(self, data_name, phase, batch_id, sequence_length=51, attribute='features'):
		if data_name not in data_names:
			print("Dataset does not exist")

		self.data_name = data_name
		self.phase = phase
		self.sequence_length = sequence_length
		self.attribute = attribute
		self.data_shape = None

		self.data = self._load_dataset(batch_id)

	def _load_dataset(self, batch_id):
		"""
		Return a dictionary of HDF5 Datasets

		keys (str): attribute
		values (h5py.Dataset): HDH5 Dataset

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
		f = h5py.File(file_path, 'r')

		# TODO: read-in large amount of data in parallel using batch
		# Warning: H5PY might not work with PyTorch Dataloader

		# shape of data is (T, B, K, W, H, C)
		data_shape = (self.sequence_length, batch_size, 1) + f[self.phase][self.attribute].shape[2:]
		self.data_shape = data_shape

		start_idx, end_idx = batch_id * batch_size, (batch_id+1) * batch_size
		data = f[self.phase][self.attribute][:self.sequence_length, start_idx:end_idx, :, :, :]

		# reshape data accordingly
		data = np.reshape(data, data_shape)

		print("Shape of loaded data:", self.data_shape)
		print()

		# remember to close file
		f.close()
		return data

	def __getitem__(self, idx):
		if idx >= self.data_shape[0]:
			return None

		d = self.data[idx, :, :, :, :, :]

		# convert data to PyTorch tensor
		t = torch.tensor(d.astype(np.float32))

		return t

	def __len__(self):
		return self.data_shape[0]

	@staticmethod
	def get_num_batches():
		return 50000 // batch_size


# train_data = Data('balls3curtain64', 'training')
# print("Number of sequences in training data:", len(train_data))
# print("Data shape", train_data)
# print("Torch size of each sequence:", train_data[0].shape)

# dataloader = DataLoader(train_data, batch_size=batch_size, 
# 						shuffle=True, num_workers=0)