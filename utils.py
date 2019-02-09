import os

import torch
import torch.nn as nn
import torchvision.transforms


def create_directory(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)


def clear_directory(dir_name, recursive=False):
	for f in os.listdir(dir_name):
		fpath = os.path.join(dir_name, f)
		try:
			if os.path.isfile(fpath):
				os.unlink(fpath)
			elif recursive and os.path.isdir(fpath):
				clear_directory(fpath, recursive)
				os.unlink(fpath)
		except Exception as e:
			print(e)


def show_image(t, b, k):
	"""
	Given an input data Tensor of shape (B, K, W, H, C),
	convert it into an image and show.
	"""
	# print(t.size())

	d = torch.squeeze(t[b][k], dim=0)
	d = d.permute(2, 1, 0)

	pil = torchvision.transforms.ToPILImage()
	im = pil(d)
	im.show()


class BCELoss(nn.Module):
	def __init__(self):
		super(BCELoss, self).__init__()

	def forward(self, y, t, use_gpu=False):
		clipped_y = torch.clamp(y, 1e-6, 1. - 1.e-6)
		res = -(t * torch.log(clipped_y) + (1. - t) * torch.log(1. - clipped_y))
		if use_gpu:
			return res.cuda()
		else:
			return res


# compute KL(p1, p2)
class KLDivLoss(nn.Module):
	def __init__(self):
		super(KLDivLoss, self).__init__()

	def forward(self, p1, p2, use_gpu=False):
		res = p1 * torch.log(torch.clamp(p1 / torch.clamp(p2, 1e-6, 1e6), 1e-6, 1e6)) + (1 - p1) * torch.log(
			torch.clamp((1 - p1) / torch.clamp(1 - p2, 1e-6, 1e6), 1e-6, 1e6))
		if use_gpu:
			return res.cuda()
		else:
			return res
