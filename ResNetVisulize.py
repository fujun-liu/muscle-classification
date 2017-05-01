'''
Muscle model test
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

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.6900, 0.3519, 0.5292], [0.1426,0.1989,0.1625])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.6900, 0.3519, 0.5292], [0.1426,0.1989,0.1625])
    ]),
}

class ResNetVisulize():
    def __init__(self):
        pass

    def _resnet_feat_extractor(self):
        return (nn.Sequential(
            self.model_ft.conv1,
            self.model_ft.bn1,
            self.model_ft.relu,
            self.model_ft.maxpool,
            self.model_ft.layer1,
            self.model_ft.layer2,
            self.model_ft.layer3,
            self.model_ft.layer4,
            self.model_ft.avgpool,
        ),)
    
    def resnet_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        conv_feat = self.layer4(x)

        x = self.avgpool(conv_feat)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, conv_feat
    
    def _resnetup_conv_feat(self):
        upsample = self.model_ft.avgpool.children()[0]
        return (nn.Sequential(
            self.model_ft.conv1,
            self.model_ft.bn1,
            self.model_ft.relu,
            self.model_ft.maxpool,
            self.model_ft.layer1,
            self.model_ft.layer2,
            self.model_ft.layer3,
            self.model_ft.layer4,
            upsample,
        ),)

    def init_atten(self, model_path, arch, test_transform=None, batch_sz=1, cuda_id=-1):
        self.cuda_id = cuda_id
        self.batch_sz = batch_sz
        self.test_transform = test_transform
        if self.cuda_id > -1:
            self.model_ft = torch.load(model_path)
            self.model_ft.cuda(device_id=cuda_id)
        else:
            self.model_ft = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model_ft.cpu()
        self.W = self.model_ft.fc.weight.data

        if 'vgg' in arch:
            self.extractor = self._vgg_feat_extractor()
        elif 'inception' in arch:
            self.extractor = self._inception_feat_extractor()
        else:
            self.extractor = self._resnet_feat_extractor()
    
    def _extract_feature(self, x):
        for nn_model in self.extractor:
            x = nn_model(x)
            x = x.view(x.size(0), -1) 
        return x.data.numpy()

    
    def extract(self, img_lst):
        '''
        img_lst is a list of images
        '''
        img0 = self.test_transform(img_lst[0])
        channels, H, W = img0.size()
        N = len(img_lst)
        X = torch.Tensor(N, channels, H, W)
        X[0] = img0
        for i in range(len(img_lst)):
            X[i] = self.test_transform(img_lst[i])
        
        proba = None
        num_batches = int(np.ceil(1.0*N/self.batch_sz))
        for b in range(num_batches):
            start = b*self.batch_sz
            this_size = self.batch_sz if b < num_batches-1 else N-b*self.batch_sz
            if self.cuda_id > -1:
                B = Variable(X[start:start+this_size,...]).cuda(self.cuda_id)
            else:
                B = Variable(X[start:start+this_size,...])
            feat_this = self._extract_feature(B)
            Feat = feat_this if Feat is None else np.concatenate((Feat, feat_this), axis=0)
        return Feat