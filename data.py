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
batch_size = 10

class Data(Dataset):
	"""
	Read in h5 data, which is consisted of 50000 sequences of 51 frames.
	
	For bouncing balls dataset, use "features" and "groups" attribute for
	training and validation.

	Attributes:
		data_name (str): name of dateset (e.g. balls4mass64)
		phase(str): "training" or "validation"
		sequence_length(int): number of frames to use
		attribute_list(tuple): list of attributes used
	"""
	def __init__(self, data_name, phase, sequence_length=51, attribute_list=('features', 'groups')):
		if data_name not in data_names:
			print("Dataset does not exist")

		self.data_name = data_name
		self.phase = phase
		self.sequence_length = sequence_length
		self.attribute_list = attribute_list
		self.data_shapes = {}
		
		self.dataset = self._load_dataset()

	def _load_dataset(self):
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

		dataset = {}

		# TODO: read-in large amount of data in parallel using batch
		# Warning: H5PY might not work with PyTorch Dataloader
		for attr in self.attribute_list:
			# shape of data is (T, B, K, W, H, C)
			data_shape = (self.sequence_length, batch_size, 1) + f[self.phase][attr].shape[2:]
			self.data_shapes[attr] = data_shape

			# reshape data accordingly
			data = f[self.phase][attr][:self.sequence_length, :batch_size, :, :, :]
			data = np.reshape(data, data_shape)
			# print(data.shape)
			dataset[attr] = data

		# print(self.data_shapes)

		# remember to close file
		f.close()
		return dataset

	def __getitem__(self, idx):
		if idx >= batch_size:
			return None

		item = {}

		for attr in self.attribute_list:
			d = self.dataset[attr][:, idx, :, :, :, :]

			# convert data to PyTorch tensor
			t = torch.Tensor(d.astype(float))

			item[attr] = t

		return item

	def __len__(self):
		for k, v in self.data_shapes.items():
			return v[1]


train_data = Data('balls3curtain64', 'training')
print("Number of sequences in training data:", len(train_data))
print("Torch size of each sequence:", train_data[0]['features'].shape)

# dataloader = DataLoader(d, batch_size=batch_size, 
# 						shuffle=False, num_workers=0)
# for i, dd in enumerate(dataloader):
# 	print(i, dd)