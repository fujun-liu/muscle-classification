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

class MuscleFeatExtractor():
    def __init__(self, scale_factor=8, num_classes=3):
        self.scale_factor = scale_factor
        self.num_classes = num_classes

    def init_train(self, num_classes, init_lr=1e-3, momentum=0.90, lr_decay_epoch=10, 
                                                   weight_decay=1e-5, cuda_id=0, arch='resnet'):
        self.num_classes = num_classes
        self.cuda_id = cuda_id
        self.lr_scheduler = LRScheduler(init_lr, lr_decay_epoch)
        self._build_model(num_classes, init_lr, momentum, weight_decay)
        
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
            model_ft.fc = nn.Linear(num_ftrs, num_classes, bias=False)
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
    
    def _vgg_feat_extractor(self):
        cls_part = list(self.model_ft.classifier.children())[:-1]
        return (self.model_ft.features, nn.Sequential(*cls_part))

    def _inception_feat_extractor(self):
        return (nn.Sequential(
            # 299 x 299 x 3
            self.model_ft.Conv2d_1a_3x3,
            # 149 x 149 x 32
            self.model_ft.Conv2d_2a_3x3,
            # 147 x 147 x 32
            self.model_ft.Conv2d_2b_3x3,
            # 147 x 147 x 64
            nn.MaxPool2d(3, stride=2),
            # 73 x 73 x 64
            self.model_ft.Conv2d_3b_1x1,
            # 73 x 73 x 80
            self.model_ft.Conv2d_4a_3x3,
            # 71 x 71 x 192
            nn.MaxPool2d(3, stride=2),
            # 35 x 35 x 192
            self.model_ft.Mixed_5b,
            # 35 x 35 x 256
            self.model_ft.Mixed_5c,
            # 35 x 35 x 288
            self.model_ft.Mixed_5d,
            # 35 x 35 x 288
            self.model_ft.Mixed_6a,
            # 17 x 17 x 768
            self.model_ft.Mixed_6b,
            # 17 x 17 x 768
            self.model_ft.Mixed_6c,
            # 17 x 17 x 768
            self.model_ft.Mixed_6d,
            # 17 x 17 x 768
            self.model_ft.Mixed_6e,
            # 17 x 17 x 768
            self.model_ft.Mixed_7a,
            # 8 x 8 x 1280
            self.model_ft.Mixed_7b,
            # 8 x 8 x 2048
            self.model_ft.Mixed_7c,
            # 8 x 8 x 2048
            nn.AvgPool2d(8)
        ),)

    def init_test(self, model_path, test_transform=None, batch_sz=1, cuda_id=-1):
        self.cuda_id = cuda_id
        self.batch_sz = batch_sz
        self.test_transform = test_transform
        if self.cuda_id > -1:
            self.model_ft = torch.load(model_path)
            self.model_ft.cuda(device_id=cuda_id)
        else:
            self.model_ft = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model_ft.cpu()
    
    def init_extract(self, model_path, arch, test_transform=None, batch_sz=1, cuda_id=-1):
        self.cuda_id = cuda_id
        self.batch_sz = batch_sz
        self.test_transform = test_transform
        if self.cuda_id > -1:
            self.model_ft = torch.load(model_path)
            self.model_ft.cuda(device_id=cuda_id)
        else:
            self.model_ft = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model_ft.cpu()
        
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
        
        Feat = None
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