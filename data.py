# import cv2
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

data_names = {'balls4mass64', 'balls678mass64', 'balls3curtain64', 'atari'}
data_path = "./data"
train_size = None
valid_size = 1000
test_size = None

def load_data(file_path, phase, out_list):
	f = h5py.File(file_path, 'r')
	data = {}
	for i in range(len(out_list)):
		d = np.array(f[phase][out_list[i]])
		data[out_list[i]] = d
	f.close()
	return data

class PhyData(Dataset):
	def __init__(self, data_name, phase, out_list=('features', 'groups')):
		if data_name not in data_names:
			print("Dataset does not exist")

		self.data_name = data_name
		self.phase = phase
		self.out_list = out_list

	def __len__(self):
		return 51

	def load_data(self):
		file_path = data_path + '/' + self.data_name + '.h5'
		self.data = load_data(file_path, self.phase, self.out_list)

	def __getitem__(self, index):
		data_dict = {self.data[dn]: ds[:self.sequence_length, start_idx:start_idx + self.batch_size][:, :, None]
			for dn, ds in self.data.items()}
		return data_dict

d = PhyData('balls3curtain64', 'training', ['features'])
d.load_data()
print(d.data.size())
# dataloader = DataLoader(d, batch_size=4, 
# 						shuffle=False, num_workers=0)
# for i, dd in enumerate(dataloader):
# 	print(i, dd)