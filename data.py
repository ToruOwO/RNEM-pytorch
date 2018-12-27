# import cv2
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# from sacred import Ingredient

# ds = Ingredient('dataset')

# @ds.config
# def cfg():
#     name = 'balls4mass64'
#     path = './data'
#     train_size = None           # subset of training set (None, int)
#     valid_size = 1000           # subset of valid set (None, int)
#     test_size = None            # subset of test set (None, int)
#     queue_capacity = 100        # nr of batches in the queue

# ds.add_named_config('balls4mass64', {'name': 'balls4mass64'})
# ds.add_named_config('balls678mass64', {'name': 'balls678mass64'})
# ds.add_named_config('balls3curtain64', {'name': 'balls3curtain64'})
# ds.add_named_config('atari', {'name': 'atari'}) 

def load_data(data_names, path):
	f = h5py.File(path, 'r')
	data = []
	for i in range(len(data_names)):
		d = np.array(f['training'][data_names[i]])
		data.append(d)
	f.close()
	return data

class PhyData(Dataset):
	def __init__(self, data_path, phase, data_names):
		self.data_path = data_path
		self.phase = phase
		self.data_names = data_names
		self.sequence_length = 10

	def __len__(self):
		return 51

	def load_data(self):
		self.data = load_data(self.data_names, self.data_path)

	def __getitem__(self, index):
		data_dict = {self.data[dn]: ds[:self.sequence_length, start_idx:start_idx + self.batch_size][:, :, None]
			for dn, ds in self.data.items()}
		return data_dict

d = PhyData('data/balls3curtain64.h5', 'training', ['features'])
d.load_data()
# dataloader = DataLoader(d, batch_size=4, 
# 						shuffle=False, num_workers=0)
# for i, dd in enumerate(dataloader):
# 	print(i, dd)