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
gid = 0
arch = 'resnet'
num_rois = 10
model_dir = 'torch-model'
model_name = 'pretrain_{}classes_cv{}{}_{}_60_{}_l1_only'.format(num_folds, num_folds, gid, arch, num_rois)
model_path = os.path.join(model_dir, model_name + '.th')
model_ft = torch.load(model_path, map_location=lambda storage, loc: storage)
model_ft.cpu()

W = np.abs(model_ft.fc.weight.data.numpy())
print len(np.nonzero(W[0] > 1e-10)[0])
print len(np.nonzero(W[1])[0])
print len(np.nonzero(W[2])[0])
#print W[0]