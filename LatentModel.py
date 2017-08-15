'''
LatentModel
'''
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
import torch.nn.functional as F
from image_sampler import ImageSampler
from PIL import Image

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

class LatentModel():
    def __init__(self, ):
        pass

    def init_train(self, num_classes, pretrained=True, init_lr=1e-3, momentum=0.90, lr_decay_epoch=10, 
                                                   weight_decay=1e-5, cuda_id=0, arch='resnet'):
        self.num_classes = num_classes
        self.cuda_id = cuda_id
        self.arch = arch
        self.pretrained = pretrained
        self.lr_scheduler = LRScheduler(init_lr, lr_decay_epoch)
        self._build_model(num_classes, init_lr, momentum, weight_decay)
        
        
    def _build_model(self, num_classes, init_lr, momentum, weight_decay):
        if self.arch == 'vgg':
            model_ft = models.vgg16(pretrained=self.pretrained)
            for idx, m in enumerate(list(model_ft.children())[1].children()):
                print(idx, '->', m)
            mylist = list(model_ft.classifier.children())
            mylist[-1] = nn.Linear(4096, num_classes)
            model_ft.classifier = nn.Sequential(*mylist)
        elif self.arch == 'inception':
            # input image size is 299 x 299
            model_ft = models.inception_v3(pretrained=self.pretrained)
            model_ft.transform_input = False
            model_ft.aux_logits = False
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft = models.resnet18(pretrained=self.pretrained)
            num_ftrs = model_ft.fc.in_features
            #model_ft.fc = nn.Linear(num_ftrs, num_classes, bias=True)
            noisy_layer = nn.Linear(num_classes+1, num_classes, bias=False)
            noisy_layer.data = torch.eye(num_classes, num_classes+1)
            model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes+1, bias=True),
                                        nn.LogSoftmax(), 
                                        noisy_layer)

        if self.cuda_id > -1:
            model_ft = model_ft.cuda(device_id=self.cuda_id)
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
        num_batches = N // self.batch_sz
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
                params = self.model_ft.parameters()
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer_ft.step()
                    # noisy weight
                    noisy_layer = list(self.model_ft.fc.children())[-1]
                    noisy_layer.weight.data = torch.clamp(noisy_layer.weight.data, .0, 1.0)
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            
            noisy_layer = list(self.model_ft.fc.children())[-1]
            print(noisy_layer.weight.data)

            epoch_loss = running_loss / self.dset_sizes[phase]
            epoch_acc = running_corrects / self.dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model = copy.deepcopy(self.model_ft)


    def train(self, dset_loaders, dset_sizes, num_epochs=25):
        self.dset_loaders = dset_loaders
        self.dset_sizes = dset_sizes
        self.num_epochs = num_epochs

        since = time.time()
        self.best_model = self.model_ft
        self.best_acc = 0.0

        for epoch in range(num_epochs):
            self._train_epoch(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        return self.best_model

if __name__ == "__main__":
    from cv_classification_dataset import CVImageFolder
    num_folds = 3
    arch = 'resnet'

    cuda_id = -1
    num_classes = 3
    init_lr, momentum, weight_decay = 1e-4, 0.95, 0.005
    num_epochs = 60
    # batch_size 8 used for previous model
    batch_size = 8
    save_model = False
    num_rois = 30
    pretrained = True
    if arch == 'inception':
        img_dir = '/media/fujunl/FujunLiu/muscle-classification/train-patches-v3-{}'.format(num_rois)
        img_suffix = '.png'
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.Scale(320),
                transforms.RandomSizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.6900, 0.3519, 0.5292], [0.1426,0.1989,0.1625])
            ]),
            'val': transforms.Compose([
                transforms.Scale(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.6900, 0.3519, 0.5292], [0.1426,0.1989,0.1625])
            ])
        }
    else:
        img_dir = '/media/fujunl/FujunLiu/muscle-classification/train-patches-{}'.format(num_rois)
        img_suffix = '.png'
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
            ])
        }

    for cv_id in range(num_folds):
        dsets = {'train':CVImageFolder(img_dir, img_suffix, transform=data_transforms['train'], cv_id=cv_id, train=True),
            'val':CVImageFolder(img_dir, img_suffix, transform=data_transforms['val'], cv_id=cv_id, train=False)}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}

        dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
        print(dset_sizes)
        muscle_model = LatentModel()
        muscle_model.init_train(num_classes, pretrained=pretrained, init_lr=init_lr, momentum=momentum, 
                                weight_decay=weight_decay, cuda_id=cuda_id, arch=arch)
        muscle_model.train(dset_loaders, dset_sizes, num_epochs=num_epochs)
        # save model
        if save_model:
            model_path = 'pretrain_{}classes_cv{}{}_{}_{}_{}_latent.th'.format(num_classes, num_folds, cv_id, 
                                                                          arch, num_epochs, num_rois)
            muscle_model.save(model_path)
        