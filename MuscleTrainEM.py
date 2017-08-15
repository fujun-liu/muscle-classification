from __future__ import print_function, division

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
from cv_classification_dataset import CVImageFolder, CVImageFolderFromList
import torch.nn.functional as F
from image_sampler import ImageSampler
from PIL import Image

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
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

class LRScheduler():
    def __init__(self, init_lr=1e-3, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        
    def __call__(self, optimizer, epoch):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = self.init_lr * (0.1**(epoch // self.lr_decay_epoch))

        if epoch % self.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

class MuscleModel():
    def __init__(self, img_dir, sample_epoches=4, scale_factor=8):
        self.scale_factor = scale_factor
        
        self.sample_epoches = sample_epoches
        

    def init_train(self, num_classes, batch_sz=8, init_lr=1e-3, momentum=0.90, lr_decay_epoch=10, 
                                                   weight_decay=1e-5, cuda_id=0, arch='resnet'):
        self.num_classes = num_classes
        self.cuda_id = cuda_id
        self.lr_scheduler = LRScheduler(init_lr, lr_decay_epoch)
        self._build_model(num_classes, init_lr, momentum, weight_decay)
        self.batch_sz = batch_sz
        
    def _build_model(self, num_classes, init_lr, momentum, weight_decay):
        if arch == 'vgg':
            model_ft = models.vgg16(pretrained=True)
            for idx, m in enumerate(list(model_ft.children())[1].children()):
                print(idx, '->', m)
            mylist = list(model_ft.classifier.children())
            mylist[-1] = nn.Linear(4096, num_classes)
            model_ft.classifier = nn.Sequential(*mylist)
        elif arch == 'inception':
            # input image size is 299 x 299
            model_ft = models.inception_v3(pretrained=True)
            model_ft.transform_input = False
            model_ft.aux_logits = False
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        elif arch == 'resnet34':
            model_ft = models.resnet34(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        elif arch == 'resnetup':
            print('feature map size:{}'.format(self.scale_factor))
            model_ft = models.resnet18(pretrained=True)
            #model_ft.avgpool = nn.Sequential(nn.ConvTranspose2d(512,512,2,stride=2), nn.AvgPool2d(14))
            model_ft.avgpool = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=self.scale_factor), 
                                             nn.AvgPool2d(7*self.scale_factor))
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes, bias=True)
        else:
            model_ft = models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes, bias=True)

        if cuda_id > -1:
            model_ft = model_ft.cuda(device_id=cuda_id)
        self.model_ft = model_ft
        
        self.criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        self.optimizer_ft = optim.SGD(self.model_ft.parameters(), 
                                          weight_decay=weight_decay, lr=init_lr, momentum=momentum)
    
    def init_test(self, model_path, test_transform=None, batch_sz=1, cuda_id=-1):
        self.cuda_id = cuda_id
        self.batch_sz = batch_sz
        self.test_transform = test_transform
        self.model_ft = torch.load(model_path)
        self.num_classes = self.model_ft.fc.out_features
        if self.cuda_id > -1:
            self.model_ft.cuda(device_id=cuda_id)

    def save(self, model_path, save_best=False):
        '''
        default is to save lasted trained model
        '''
        if save_best:
            #for idx, m in enumerate(self.best_model.children()):
            #    print (idx, '->', m)
            torch.save(self.best_model, model_path)
        else:
            #for idx, m in enumerate(self.model_ft.children()):
            #    print (idx, '->', m)
            torch.save(self.model_ft, model_path)

    def pred_proba(self, img_lst):
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
        
        proba = torch.Tensor(N, self.num_classes)
        num_batches = int(np.ceil(1.0*N/self.batch_sz))
        for b in range(num_batches):
            start = b*self.batch_sz
            this_size = self.batch_sz if b < num_batches-1 else N-b*self.batch_sz
            if self.cuda_id > -1:
                B = Variable(X[start:start+this_size,...]).cuda(self.cuda_id)
            else:
                B = Variable(X[start:start+this_size,...])
            proba[start:start+this_size,:] = F.softmax(self.model_ft(B)).data
        return proba.numpy()
    
    def pred_proba_from_tensor(self, X):
        N = X.size()[0]
        proba = torch.Tensor(N, self.num_classes)
        num_batches = int(np.ceil(1.0*N/self.batch_sz))
        for b in range(num_batches):
            start = b*self.batch_sz
            this_size = self.batch_sz if b < num_batches-1 else N-b*self.batch_sz
            if self.cuda_id > -1:
                B = Variable(X[start:start+this_size,...]).cuda(self.cuda_id)
            else:
                B = Variable(X[start:start+this_size,...])
            proba[start:start+this_size,:] = F.softmax(self.model_ft(B)).data
        return proba.numpy()

    def pred(self, img_lst):
        proba = self.pred_proba(img_lst)
        return np.argmax(proba, axis=1)


    def _train_epoch(self, epoch):
        print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                self.optimizer_ft = self.lr_scheduler(self.optimizer_ft, epoch)
                self.model_ft.train(True)  # Set model to training mode
            else:
                self.model_ft.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in self.dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if self.cuda_id > -1:
                    inputs = Variable(inputs).cuda(self.cuda_id)
                    labels = Variable(labels).cuda(self.cuda_id)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                self.optimizer_ft.zero_grad()

                # forward
                outputs = self.model_ft(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer_ft.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / self.dset_sizes[phase]
            epoch_acc = running_corrects / self.dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model = copy.deepcopy(self.model_ft)

    def _load_train_data(self, img_lst):
        N = len(img_lst)
        img0 = self.test_transform(Image.open(img_lst[0]))
        channels, H, W = img0.size()
        N = len(img_lst)
        self.trainX = torch.Tensor(N, channels, H, W)
        for i in range(N):
            self.trainX[i] = self.test_transform(Image.open(img_lst[i]))

    def train(self, cv_id, num_epochs=25):
        self.num_epochs = num_epochs

        since = time.time()
        self.best_model = self.model_ft
        self.best_acc = 0.0
        self.test_transform = data_transforms['val']

        img_lst_all = glob.glob(self.img_dir + '/*.png')
        img_lst_train = [img_path for img_path in img_lst_all if int(img_path[-7]) != cv_id]
        img_lst_test = [img_path for img_path in img_lst_all if int(img_path[-7]) == cv_id]
        self._load_train_data(img_lst_train)
        self.sampler = ImageSampler(img_lst_train)
        
        self.dset_loaders = {}
        self.dset_sizes = {}
        dsets_val = ImageFolderFromList(img_lst_test, transform=data_transforms['val'])
        self.dset_loaders['val'] = torch.utils.data.DataLoader(dsets_val, batch_size=self.batch_sz, shuffle=True, num_workers=1)
        self.dset_sizes['val'] = len(dsets_val)
        for epoch in range(num_epochs):
            if epoch%self.sample_epoches == 0:
                weight = None
                if epoch > 0:
                    weight = self.pred_proba_from_tensor(self.trainX)
                img_lst_sample = self.sampler.sample(weight=weight)
                dsets_train = ImageFolderFromList(img_lst_sample, transform=data_transforms['train'])
                self.dset_loaders['train'] = torch.utils.data.DataLoader(dsets_train, batch_size=self.batch_sz, shuffle=True, num_workers=1)
                self.dset_sizes['train'] = len(dsets_train)
            self._train_epoch(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        return self.best_model

