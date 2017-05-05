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

crop_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224)
    ])

class ResNetVisulize():
    def __init__(self):
        pass
    
    def resnet_forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        conv_feat = self.model_ft.layer4(x)

        x = self.model_ft.avgpool(conv_feat)
        x = x.view(x.size(0), -1)
        x = self.model_ft.fc(x)
        return x.data.numpy(), conv_feat.data.numpy()
    
    def resnetup_forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)

        uplayer, avg_layer = list(self.model_ft.avgpool.children())
        conv_feat = uplayer(x)
        x = avg_layer(conv_feat)
        x = x.view(x.size(0), -1)
        x = self.model_ft.fc(x)
        return x.data.numpy(), conv_feat.data.numpy()

    def resnetfine_forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)

        uplayer, conv_layer, avg_layer = list(self.model_ft.avgpool.children())
        x = uplayer(x)
        conv_feat = conv_layer(x)
        x = avg_layer(conv_feat)
        x = x.view(x.size(0), -1)
        x = self.model_ft.fc(x)
        return x.data.numpy(), conv_feat.data.numpy()

    def init_atten(self, model_path, arch, test_transform=None, atten_type='cam', batch_sz=1, cuda_id=-1):
        self.cuda_id = cuda_id
        self.batch_sz = batch_sz
        self.test_transform = test_transform
        if self.cuda_id > -1:
            self.model_ft = torch.load(model_path)
            self.model_ft.cuda(device_id=cuda_id)
        else:
            self.model_ft = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model_ft.cpu()
        self.arch = arch
        self.atten_type = atten_type
        self.W = self.model_ft.fc.weight.data.numpy()
    
    
    def compute_cam(self, img_lst, conv_feat_all, proba_all):
        atten_lst = []
        crop_lst = []
        max_val = .0
        slide_pred = np.argmax(np.mean(proba_all, axis=0))
        N = len(img_lst)
        for i in range(N):
            pred = np.argmax(proba_all[i])
            atten_map = conv_feat_all[i] * self.W[pred][:,np.newaxis,np.newaxis]
            atten_map = np.sum(atten_map, axis=0)
            atten_map[atten_map < .0] = .0
            max_val = max(max_val, np.amax(atten_map))
            atten_lst.append(atten_map)
            crop_lst.append(np.asarray(crop_transforms(img_lst[i])))
        return crop_lst, atten_lst, proba_all, max_val
    
    def compute_top_neuron(self, img_lst, conv_feat_all, proba_all):
        slide_pred = np.argmax(np.mean(proba_all, axis=0))
        gap = np.mean(conv_feat_all, axis=(2,3))
        neurons = gap * self.W[slide_pred][np.newaxis,:]
        neurons = np.sum(neurons, axis=0)
        indice = np.argsort(neurons)[::-1]
        top_index = indice[0]

        atten_lst = []
        crop_lst = []
        max_val = .0
        N = len(img_lst)
        for i in range(N):
            atten_map = conv_feat_all[i, top_index]
            atten_map[atten_map < .0] = .0
            max_val = max(max_val, np.amax(atten_map))
            atten_lst.append(atten_map)
            crop_lst.append(np.asarray(crop_transforms(img_lst[i])))
        return crop_lst, atten_lst, proba_all, max_val

    def compute_attention(self, img_lst):
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
        
        proba_all = None
        conv_feat_all = None
        num_batches = int(np.ceil(1.0*N/self.batch_sz))
        for b in range(num_batches):
            start = b*self.batch_sz
            this_size = self.batch_sz if b < num_batches-1 else N-b*self.batch_sz
            if self.cuda_id > -1:
                B = Variable(X[start:start+this_size,...]).cuda(self.cuda_id)
            else:
                B = Variable(X[start:start+this_size,...])
            if 'fine' in self.arch:
                proba_this, conv_feat_this = self.resnetfine_forward(B)
            elif 'up' in self.arch:
                proba_this, conv_feat_this = self.resnetup_forward(B)
            else:
                proba_this, conv_feat_this = self.resnet_forward(B)
            proba_all = proba_this if proba_all is None else np.concatenate((proba_all, proba_this), axis=0)
            conv_feat_all = conv_feat_this if conv_feat_all is None else np.concatenate((conv_feat_all, conv_feat_this), axis=0)
        
        if self.atten_type == 'cam':
            return self.compute_cam(img_lst, conv_feat_all, proba_all)
        else:
            return self.compute_top_neuron(img_lst, conv_feat_all, proba_all)

