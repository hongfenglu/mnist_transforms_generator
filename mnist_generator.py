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

# num_batch=70

train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=1, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(ds['test'], batch_size=1, shuffle=False, num_workers=4)

# sample exactly 50 numbers from each class

print('training size: ', len(ds['train']))
print('test size: ', len(ds['test']))

counts = np.zeros(10)
train_data = torch.Tensor([])
train_target = torch.LongTensor([])
# # create dictionary of mnist by class
for i, (data, target) in enumerate(train_loader):

	if i < 1000:
		continue
	t = target.item()
	if np.all(counts==50):
		break
	if counts[t] >= 50:
		continue
	counts[t] += 1
	train_data = torch.cat((train_data, data), 0)
	train_target = torch.cat((train_target, target), 0)

z = list(zip(train_data, train_target.detach().numpy()))
print(len(z))

ds['train'] = z
train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=10, shuffle=False, num_workers=4)


'''
	Rotation: -45, 0, 45
	Scale: 4 values linearly spaced in [1, 1.9]
	Position X: 5 values in [-0.3, 0.3]
	Position Y: 5 values in [-0.3, 0.3]
	We varied one latent at a time (starting from Position Y, then Position X, etc), 
	and sequentially stored the images in fixed order.
'''

def get_all_transformations(data, no_corner=False):
	transformed = torch.Tensor([])
	for scaler in scalers:
		s = batch_compatible(scaler)
		trans = s(data)
		transformed = torch.cat((transformed, trans), 0)
	data = transformed
	transformed = torch.Tensor([])
	for rotator in rotators:
		r = batch_compatible(rotator)
		trans = r(data)
		transformed = torch.cat((transformed, trans), 0)
	
	data = transformed
	train = torch.Tensor([])
	if not no_corner:
		for translator in translators:
			t = batch_compatible(translator)
			trans = t(data)
			train = torch.cat((train, trans), 0)
		return train
	else:
		holdout = torch.Tensor([])
		for i, translator in enumerate(translators):
			t = batch_compatible(translator)
			trans = t(data)
			if i in [0, 1, 5, 6]:
				holdout = torch.cat((holdout, trans), 0)
			else:
				train = torch.cat((train, trans), 0)
		return train, holdout



rotate_n = 3
# rotate_deg = 360//rotate_n
rotators = [Rotate(360-45)(), Rotate(0)(), Rotate(45)()]

scale_n = 3
scalers = [Scale(1.4+0.3*i)() for i in range(scale_n)]

translate_n = 5
x = [-0.3, -0.15, 0, 0.15, 0.3]
pos = list(itertools.product(x, x))
translators = [Translate(*i)() for i in pos]

def generate_data(folder, method):
	all_data = OrderedDict()
	all_data['train'] = []
	all_data['holdout'] = []
	num_of_trans =  rotate_n*scale_n*translate_n**2

	# import ipdb; ipdb.set_trace()
	if method == 'position_all':
		p = 0.1
		for i, (data, target) in enumerate(train_loader):
			
			data = place_subimage_in_background((64, 64))(data)
			train = get_all_transformations(data)
			train_target = target.repeat(num_of_trans)
			z = list(zip(train, train_target.detach().numpy()))
			n = len(z)
			# import ipdb; ipdb.set_trace()

			holdout_indices = random.sample(range(n), int(n*p))
			train_indices = [i for i in range(n) if i not in holdout_indices]

			train_z = list(itemgetter(*train_indices)(z))
			holdout_z = list(itemgetter(*holdout_indices)(z))

			# import ipdb; ipdb.set_trace()
			all_data['train'].extend(train_z)
			all_data['holdout'].extend(holdout_z)

			print('finish ', i, ' ', 'train len: ', len(all_data['train']))

		# n = all_data['train']
		# p = 0.1
		# random_indices = random.sample(range(n), int(n*p))
		# all_data['holdout'] = list(np.array(all_data['train'])[ramdom_indices])
		# with open(os.path.join(outdir, folder+'.pickle'), 'wb') as output:
		# 	pickle.dump(all_data, output)

		if not os.path.exists(os.path.join(outdir, folder)):
			os.makedirs(os.path.join(outdir, folder))
			for i in range(10):
				os.makedirs(os.path.join(outdir, folder, 'train', str(i)))
				os.makedirs(os.path.join(outdir, folder, 'holdout', str(i)))

		for k in range(len(all_data['train'])):
			save_image(all_data['train'][k][0], os.path.join(outdir, folder, 'train', str(all_data['train'][k][1]), str(k) + '.png'))
				
		for k in range(len(all_data['holdout'])):
			save_image(all_data['holdout'][k][0], os.path.join(outdir, folder, 'holdout', str(all_data['holdout'][k][1]), str(k) + '.png'))
		
	elif method == 'position_no_corner':
		# missing the lower right 2by2 positions (-0.3, -0.3), (-0.3, -0.15), (-0.15, -0.3), (-0.15, -0.15)
		# translator[0, 1, 5, 6]
		# need 10 separate datasets, each missing the corner for one class
		all_holdout_target = np.array([])

		for i, (data, target) in enumerate(train_loader):
		
			data = place_subimage_in_background((64, 64))(data)
			train, holdout = get_all_transformations(data, no_corner=True)
			train_target = target.repeat(num_of_trans - 4*rotate_n*scale_n)
			holdout_target = target.repeat(4*rotate_n*scale_n).detach().numpy()
			all_holdout_target = np.append(all_holdout_target, holdout_target)

			# import ipdb; ipdb.set_trace()
			train_z = list(zip(train, train_target.detach().numpy()))
			holdout_z = list(zip(holdout, holdout_target))

			all_data['train'].extend(train_z)
			all_data['holdout'].extend(holdout_z)	# all data at corner
			print('finish ', i, ' ', 'train len: ', len(all_data['train']))

		# all_holdout_target = all_holdout_target.detach().numpy()

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
			print('holdout size for '+ str(i) + ': ', len(data_by_class[i]['holdout']))

		# if not os.path.exists(os.path.join(outdir, folder)):
		# 	os.makedirs(os.path.join(outdir, folder))
		# for i in range(10):
		# 	with open(os.path.join(outdir, folder, 'missing_'+str(i)+'.pickle'), 'wb') as output:
		# 		pickle.dump(data_by_class[i], output)

		if not os.path.exists(os.path.join(outdir, folder)):
			os.makedirs(os.path.join(outdir, folder))
			for i in range(10):
				os.makedirs(os.path.join(outdir, folder, 'missing_'+str(i)))
				os.makedirs(os.path.join(outdir, folder, 'missing_0', 'train', str(i)))
				# os.makedirs(os.path.join(outdir, folder, 'missing_'+str(i), 'holdout'))
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
	folder = "position_all_new"	# position_no_corner, position_all
	method = "position_all"	# 
	# train_path = '{}_train'.format(path)
	# test_path = '{}_test'.format(path)
	# valid_path = '{}_valid'.format(path)


	generate_data(folder, method)

# python3 mnist_transforms/mnist_generator.py

