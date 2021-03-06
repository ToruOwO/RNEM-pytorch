import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.animation as animation


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


def get_gamma_colors(nr_colors):
	hsv_colors = np.ones((nr_colors, 3))
	hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2 / 3) % 1.0
	color_conv = hsv_to_rgb(hsv_colors)
	return color_conv


def overview_plot(i, gammas, preds, inputs, corrupted=None, **kwargs):
	# Note: all inputs are torch tensors and must have been converted to numpy arrays

	T, B, K, W, H, C = gammas.shape
	T -= 1  # remove initialization step

	corrupted = corrupted if corrupted is not None else inputs
	gamma_colors = get_gamma_colors(K)

	# only use data in the dimension of sample i
	inputs = inputs[:, i, 0]
	gammas = gammas[:, i, :, :, :, 0]
	if preds.shape[1] != B:
		preds = preds[:, 0]
	preds = preds[:, i]
	corrupted = corrupted[:, i, 0]

	inputs = inputs.detach().numpy()
	gammas = gammas.detach().numpy()
	preds = preds.detach().numpy()
	corrupted = corrupted.detach().numpy()

	inputs = np.clip(inputs, 0., 1.)
	preds = np.clip(preds, 0., 1.)
	corrupted = np.clip(corrupted, 0., 1.)

	def plot_img(ax, data, cmap='Greys_r', xlabel=None, ylabel=None):
		if data.shape[-1] == 1:
			ax.matshow(data[:, :, 0], cmap=cmap, vmin=0., vmax=1., interpolation='nearest')
		else:
			ax.imshow(data, interpolation='nearest')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel(xlabel, color='k') if xlabel else None
		ax.set_ylabel(ylabel, color='k') if ylabel else None

	def plot_gamma(ax, gamma, xlabel=None, ylabel=None):
		gamma = np.transpose(gamma, [1, 2, 0])
		gamma = gamma.reshape(-1, gamma.shape[-1]).dot(gamma_colors).reshape(gamma.shape[:-1] + (3,))
		ax.imshow(gamma, interpolation='nearest')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel(xlabel) if xlabel else None
		ax.set_ylabel(ylabel) if ylabel else None

	nrows, ncols = (K + 4, T + 1)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

	axes[0, 0].set_visible(False)
	axes[1, 0].set_visible(False)
	plot_gamma(axes[2, 0], gammas[0], ylabel='Gammas')
	for k in range(K + 1):
		axes[k + 3, 0].set_visible(False)
	for t in range(1, T + 1):
		g = gammas[t]
		p = preds[t]

		reconst = np.sum(g[:, :, :, None] * p, axis=0)
		plot_img(axes[0, t], inputs[t])
		plot_img(axes[1, t], reconst)
		plot_gamma(axes[2, t], g)
		for k in range(K):
			plot_img(axes[k + 3, t], p[k], ylabel=('mu_{}'.format(k) if t == 1 else None))

		plot_img(axes[K + 3, t], corrupted[t - 1])
	plt.subplots_adjust(hspace=0.1, wspace=0.1)
	return fig


def color_spines(ax, color, lw=2):
	for sn in ['top', 'bottom', 'left', 'right']:
		ax.spines[sn].set_linewidth(lw)
		ax.spines[sn].set_color(color)
		ax.spines[sn].set_visible(True)


def overview_gif(name, i, nr_steps, rollout_steps, gammas, preds, inputs, corrupted=None):
	T, B, K, W, H, C = gammas.shape
	T -= 1  # the initialization doesn't count as iteration
	gamma_colors = get_gamma_colors(K)

	# restrict to sample i and get rid of useless dims
	inputs = inputs[:, i, 0]
	gammas = gammas[:, i, :, :, :, 0]
	if preds.shape[1] != B:
		preds = preds[:, 0]
	preds = preds[:, i]

	inputs = inputs.detach().numpy()
	gammas = gammas.detach().numpy()
	preds = preds.detach().numpy()

	inputs = np.clip(inputs, 0., 1.)
	preds = np.clip(preds, 0., 1.)

	def plot_img(ax, data, cmap='Greys_r', xlabel=None, ylabel=None, border_color=None):
		if data.shape[-1] == 1:
			ax.matshow(data[:, :, 0], cmap=cmap, vmin=0., vmax=1., interpolation='nearest')
		else:
			ax.imshow(data, interpolation='nearest')
		ax.set_xticks([]); ax.set_yticks([])
		ax.set_xlabel(xlabel, color=border_color or 'k') if xlabel else None
		ax.set_ylabel(ylabel, color=border_color or 'k') if ylabel else None
		if border_color:
			color_spines(ax, color=border_color)

	def plot_gamma(ax, gamma, xlabel=None, ylabel=None):
		gamma = ((gamma - gamma.min()) / (gamma.max() - gamma.min())) * 255
		gamma = np.transpose(gamma, [1, 2, 0])
		gamma = gamma.reshape(-1, gamma.shape[-1]).dot(gamma_colors).reshape(gamma.shape[:-1] + (3,))
		ax.imshow(gamma, interpolation='nearest')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel(xlabel) if xlabel else None
		ax.set_ylabel(ylabel) if ylabel else None

	fig, axes = plt.subplots(nrows=3, figsize=(2, 6))
	fig.suptitle(name + '_{}'.format(i))

	def update(t):
		label = 'step {}'.format(t)

		g = gammas[t + 1]
		p = preds[t + 1]

		# blue border if it's still observation
		# green border otherwise
		if t < nr_steps:
			border_color = 'b'

			# ground truth
			reconst = inputs[t]
		else:
			border_color = 'g'

			# rollout
			reconst = np.sum(g[:, :, :, None] * p, axis=0)
			reconst = ((reconst - reconst.min()) / (reconst.max() - reconst.min()))

			# truth = reconst
			# truth[truth > 0.1] = 1.0
			# truth[truth <= 0.1] = 0.0
			# np.savetxt("truth_{}.csv".format(t), truth[:, :, 0], delimiter=",")
			# truth = truth[:, :, 0]
			#
			# from PIL import Image
			# im = Image.fromarray(truth*255).convert('RGB')
			# im.save("truth_{}.png".format(t))

		# debug output
		# if t == 0 or t == 6:
		# 	np.savetxt("rollout_{}.csv".format(t), reconst[:, :, 0], delimiter=",")
		# 	np.savetxt("gamma_{}.csv".format(t), g[0], delimiter=",")
		# 	np.savetxt("pred_{}.csv".format(t), p[0, :, :, 0], delimiter=",")

		plot_img(axes[0], inputs[t], ylabel='GT')
		plot_img(axes[1], reconst, ylabel='rollout', border_color=border_color)
		plot_gamma(axes[2], g, xlabel=label, ylabel='grouping')

		# must return a tuple
		return (axes,)

	# plt.subplots_adjust(hspace=0.2, wspace=0.2)

	anim = animation.FuncAnimation(fig, update, frames=np.arange(nr_steps+rollout_steps), interval=200)
	plt.show()

	# save using imagemick
	print("saving rollout overview " + name + '_{}.gif'.format(i))
	anim.save(name + '_{}.gif'.format(i), writer='imagemagick')

	return fig


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
	im.save("test_{}_{}.jpg".format(b, k), "JPEG")


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
