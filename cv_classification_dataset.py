'''
This is an implementation of fcn dataset
'''
import torch.utils.data as data
from PIL import Image
import os, glob
import torch

class CVImageFolder(data.Dataset):

    def __init__(self, img_dir, img_suffix, transform=None, target_transform=None, cv_id=0, train=True):
        img_lst_all = glob.glob(img_dir + '/*' + img_suffix)
        if len(img_lst_all) == 0:
            raise(RuntimeError("No images found. Please check dataset input."))
        if train:
            img_lst = [img_path for img_path in img_lst_all if int(img_path[-7]) != cv_id]
        else:
            img_lst = [img_path for img_path in img_lst_all if int(img_path[-7]) == cv_id]
        
        self.img_lst = img_lst
        self.targets = [int(img_path[-5]) for img_path in img_lst]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.img_lst[index]).convert('RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_lst)

class CVImageFolderFromList(data.Dataset):
    
    def __init__(self, img_lst_all, transform=None, target_transform=None, cv_id=0, train=True):
        
        if train:
            img_lst = [img_path for img_path in img_lst_all if int(img_path[-7]) != cv_id]
        else:
            img_lst = [img_path for img_path in img_lst_all if int(img_path[-7]) == cv_id]
        
        self.img_lst = img_lst
        self.targets = [int(img_path[-5]) for img_path in img_lst]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.img_lst[index]).convert('RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_lst)

class ImageFolderFromList(data.Dataset):
    
    def __init__(self, img_lst, transform=None, target_transform=None):
        
        self.img_lst = img_lst
        self.targets = [int(img_path[-5]) for img_path in img_lst]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.img_lst[index]).convert('RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_lst)
