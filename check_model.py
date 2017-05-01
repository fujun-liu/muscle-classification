'''
check model
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import torch.nn.functional as F

num_folds = 3
gid = 2
arch = 'resnet'
num_rois = 10
model_dir = 'torch-model'
model_name = 'pretrain_{}classes_cv{}{}_{}_60_{}'.format(num_folds, num_folds, gid, arch, num_rois)
model_path = os.path.join(model_dir, model_name + '.th')
model_ft = torch.load(model_path, map_location=lambda storage, loc: storage)
model_ft.cpu()

W = model_ft.fc.weight.data.numpy()
weight = np.sum(W*W, axis=0)
print np.mean(weight), np.amin(weight)
print len(np.nonzero(weight > 1e-4)[0])