from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchsample
from image_transforms import *
from collections import OrderedDict
import itertools
import pickle
import random
from operator import itemgetter
from torchvision.utils import save_image
import shutil

init_seed = 0
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)
random.seed(init_seed)

outdir = 'mnist_datasets'

ds = OrderedDict()
ds['train'] = datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
ds['test'] = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)


train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=100, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(ds['test'], batch_size=100, shuffle=False, num_workers=4)


def get_all_transformations(data, holdout_scale=0, all_scale=True):

	train = torch.Tensor([])
	holdout = torch.Tensor([])

	for i, scaler in enumerate(scalers):
		s = batch_compatible(scaler)
		trans = s(data)
		if not all_scale and i==holdout_scale:
			holdout = torch.cat((holdout, trans), 0)
		else:
			train = torch.cat((train, trans), 0)
	
	return train, holdout


scales = [1, 1.6, 2.2]
scale_n = 3
scalers = [Scale(i)() for i in scales]


def generate_data(folder, method):
	all_data = OrderedDict()
	all_data['train'] = []
	all_data['holdout'] = []
	num_of_trans =  scale_n

	if method == 'scale_all_no_big':
		for i, (data, target) in enumerate(train_loader):
			if i > 250:
				break
			# data = place_subimage_in_background((64, 64))(data)
			train, holdout = get_all_transformations(data, all_scale=False)
			train_target = target.repeat(num_of_trans - 1)
			holdout_target = target.repeat(1).detach().numpy()
			# all_holdout_target = np.append(all_holdout_target, holdout_target)

			# import ipdb; ipdb.set_trace()
			train_z = list(zip(train, train_target.detach().numpy()))
			holdout_z = list(zip(holdout, holdout_target))

			all_data['train'].extend(train_z)
			all_data['holdout'].extend(holdout_z)	# all data at corner
			print('finish ', i, ' ', 'train len: ', len(all_data['train']))

		if not os.path.exists(os.path.join(outdir, folder)):
			os.makedirs(os.path.join(outdir, folder))
			for i in range(10):
				os.makedirs(os.path.join(outdir, folder, 'train', str(i)))
				os.makedirs(os.path.join(outdir, folder, 'holdout', str(i)))

		for k in range(len(all_data['train'])):
			save_image(all_data['train'][k][0], os.path.join(outdir, folder, 'train', str(all_data['train'][k][1]), str(k) + '.png'))
				
		for k in range(len(all_data['holdout'])):
			save_image(all_data['holdout'][k][0], os.path.join(outdir, folder, 'holdout', str(all_data['holdout'][k][1]), str(k) + '.png'))
		
	elif method == 'scale_all':
		p = 0.1
		for i, (data, target) in enumerate(train_loader):
			if i > 300:
				break

			# data = place_subimage_in_background((64, 64))(data)
			train = get_all_transformations(data, all_scale=True)[0]
			train_target = target.repeat(num_of_trans)
			z = list(zip(train, train_target.detach().numpy()))
			n = len(z)
			# import ipdb; ipdb.set_trace()

			all_data['train'].extend(z)

			print('finish ', i, ' ', 'train len: ', len(all_data['train']))
		
		for i, (data, target) in enumerate(test_loader):
			if i > 25:
				break
			# data = place_subimage_in_background((64, 64))(data)
			test = get_all_transformations(data, all_scale=True)[0]
			test_target = target.repeat(num_of_trans)
			z = list(zip(test, test_target.detach().numpy()))
			n = len(z)

			all_data['holdout'].extend(z)

		if not os.path.exists(os.path.join(outdir, folder)):
			os.makedirs(os.path.join(outdir, folder))
			for i in range(10):
				os.makedirs(os.path.join(outdir, folder, 'train', str(i)))
				os.makedirs(os.path.join(outdir, folder, 'holdout', str(i)))

		for k in range(len(all_data['train'])):
			save_image(all_data['train'][k][0], os.path.join(outdir, folder, 'train', str(all_data['train'][k][1]), str(k) + '.png'))
				
		for k in range(len(all_data['holdout'])):
			save_image(all_data['holdout'][k][0], os.path.join(outdir, folder, 'holdout', str(all_data['holdout'][k][1]), str(k) + '.png'))
		
	elif method == 'scale_no_small':
		# missing the lower right 2by2 positions (-0.3, -0.3), (-0.3, -0.15), (-0.15, -0.3), (-0.15, -0.15)
		# translator[0, 1, 5, 6]
		# need 10 separate datasets, each missing the corner for one class
		all_holdout_target = np.array([])

		for i, (data, target) in enumerate(train_loader):
			if i > 250:
				break
			# data = place_subimage_in_background((64, 64))(data)
			train, holdout = get_all_transformations(data, all_scale=False)
			train_target = target.repeat(num_of_trans - 1)
			holdout_target = target.repeat(1).detach().numpy()
			all_holdout_target = np.append(all_holdout_target, holdout_target)

			# import ipdb; ipdb.set_trace()
			train_z = list(zip(train, train_target.detach().numpy()))
			holdout_z = list(zip(holdout, holdout_target))

			all_data['train'].extend(train_z)
			all_data['holdout'].extend(holdout_z)	# all data at corner
			print('finish ', i, ' ', 'train len: ', len(all_data['train']))


		data_by_class = OrderedDict()
		for i in range(10):
			data_by_class[i] = OrderedDict()
			# data_by_class[i]['train'] = all_data['train'].copy()

			holdout_indices = np.where(all_holdout_target == i)[0].tolist()
			rest_indices = np.where(all_holdout_target != i)[0].tolist()
			# import ipdb; ipdb.set_trace()

			data_by_class[i]['holdout'] = list(itemgetter(*holdout_indices)(all_data['holdout']))	# class i at corner
			data_by_class[i]['rest_train'] = list(itemgetter(*rest_indices)(all_data['holdout']))	# rest of the classes at corner
			# print('train size for '+ str(i) + ': ', len(data_by_class[i]['train']))
			# print('holdout size for '+ str(i) + ': ', len(data_by_class[i]['holdout']))

	
		if not os.path.exists(os.path.join(outdir, folder)):
			os.makedirs(os.path.join(outdir, folder))
			for i in range(10):
				os.makedirs(os.path.join(outdir, folder, 'missing_'+str(i)))
				os.makedirs(os.path.join(outdir, folder, 'missing_0', 'train', str(i)))
				os.makedirs(os.path.join(outdir, folder, 'missing_0', 'holdout', str(i)))

		for k in range(len(all_data['train'])):
			save_image(all_data['train'][k][0], os.path.join(outdir, folder, 'missing_0', 'train', str(all_data['train'][k][1]), str(k) + '.png'))
		
		srcDir_train = os.path.join(outdir, folder, 'missing_0', 'train')
		srcDir_holdout = os.path.join(outdir, folder, 'missing_0', 'holdout')
		for i in range(1, 10):
			shutil.copytree(srcDir_train, os.path.join(outdir, folder, 'missing_'+str(i), 'train'), symlinks=False, ignore=None)
			shutil.copytree(srcDir_holdout, os.path.join(outdir, folder, 'missing_'+str(i), 'holdout'), symlinks=False, ignore=None)

		for i in range(10):
			for k in range(len(data_by_class[i]['rest_train'])):
				save_image(data_by_class[i]['rest_train'][k][0], \
					os.path.join(outdir, folder, 'missing_'+str(i),'train', str(data_by_class[i]['rest_train'][k][1]), str(k+112500) + '.png'))
					
			for k in range(len(data_by_class[i]['holdout'])):
				save_image(data_by_class[i]['holdout'][k][0], \
					os.path.join(outdir, folder, 'missing_'+str(i), 'holdout', str(data_by_class[i]['holdout'][k][1]), str(k) + '.png'))




if __name__ == "__main__":
	folder = "3scale_no_big"	# position_no_corner, position_all
	method = "scale_no_small"	# 
	# train_path = '{}_train'.format(path)
	# test_path = '{}_test'.format(path)
	# valid_path = '{}_valid'.format(path)


	generate_data(folder, method)

