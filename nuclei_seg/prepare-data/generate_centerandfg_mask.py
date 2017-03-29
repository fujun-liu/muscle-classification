from scipy.io import loadmat
from scipy import ndimage
import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import cv2
from skimage.measure import regionprops

def generate_masks():
    
    gt_paths = glob.glob(gt_dir + '/*' + gt_ext)
    for gt_path in gt_paths:
        print 'generate masks for {}'.format(gt_path)
        img_name = os.path.split(gt_path)[-1][:-len(gt_ext)]
        xy = np.asarray(loadmat(gt_path)['cents']).astype(int)
        img_path = os.path.join(img_dir, img_name + img_ext)
        img = np.asarray(cv2.imread(img_path))
        imgH, imgW, _ = img.shape
        center_mask = np.zeros([imgH, imgW])
        center_mask[xy[:,1], xy[:,0]] = 1
        print len(xy)
        print  np.sum(center_mask)
        dist = ndimage.distance_transform_edt(1 - center_mask)
        weight_map = np.exp(-dist**2/sigma**2)
        print np.max(weight_map)
        dst_path = os.path.join(annot_weight_dir, img_name + '_weight.png')
        cv2.imwrite(dst_path, weight_map*255)
        dst_path = os.path.join(annot_weight_dir, img_name + '_annot.png')
        cv2.imwrite(dst_path, center_mask*255)
        


img_dir = '/media/fujunl/FujunLiu/muscle-classification/nuclei_annotation/Images'
gt_dir = '/media/fujunl/FujunLiu/muscle-classification/nuclei_annotation/positiveCoords'
annot_weight_dir = 'tmp'
img_ext, gt_ext = '.bmp', '.mat'
sigma = 3

if __name__ == '__main__':
	generate_masks()


