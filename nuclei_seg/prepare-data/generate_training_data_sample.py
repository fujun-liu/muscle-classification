import os,sys
import glob,shutil
import numpy as np
import random
from PIL import Image, ImageDraw
import cv2

img_dir, img_ext = '/media/fujunl/FujunLiu/muscle-classification/nuclei_annotation/Images', '.bmp'
weight_dir, weight_ext = 'tmp', '_weight.png'
annot_dir, annot_ext = 'tmp', '_annot.png'
patch_size = 100
stride = 50
train_dir = 'train'

def generate_training_data_sliding():
	'''
	Generate training samples for training in sliding window way from a big Image
	'''
	center_paths = glob.glob(weight_dir + '/*' + weight_ext)
	R = patch_size // 2
	rot_angles = [0,1,2]
	random.seed()
	for center_path in center_paths[:1]:
	    print 'Generating patches from {}'.format(center_path)

        img_name = os.path.split(center_path)[-1][:-len(weight_ext)]
        img_path = os.path.join(img_dir, img_name + img_ext)
        weight_map = cv2.imread(center_path)
        img = cv2.imread(img_path)
        imgH, imgW = img.shape[:2]
	    # get all patch coords
        start_x = R + random.randint(0, stride-1)
        start_y = R + random.randint(0, stride-1)
        gridX, gridY = range(start_x, imgW-R, stride), range(start_y, imgH-R, stride)   
        [meshY, meshX] = np.meshgrid(gridY, gridX)
        meshY = meshY.flatten()
        meshX = meshX.flatten() 
        #print sample_indice
        for iy, ix in zip(meshY, meshX):
            patch_img = img[iy-R+1:iy+R+1, ix-R+1:ix+R+1,:]
            patch_weight = weight_map[iy-R+1:iy+R+1, ix-R+1:ix+R+1]
            for rotk in rot_angles: 
                patch_img_rot = np.rot90(patch_img, rotk)
                patch_weight_rot = np.rot90(patch_weight, rotk)
                pid = '{}_{}_{}'.format(iy, ix, rotk)
                patch_img_dst = os.path.join(train_dir, img_name + '_' + pid + '.png')
                cv2.imwrite(patch_img_dst, patch_img_rot)
                patch_weight_dst = os.path.join(train_dir, img_name + '_' + pid + '_weight.png')
                cv2.imwrite(patch_weight_dst, patch_weight_rot)

def generate_training_data_sampling():
    	'''
	Generate training samples centering around each centers
	'''
	center_paths = glob.glob(weight_dir + '/*' + weight_ext)
	R = patch_size // 2
	rot_angles = [0,1,2]
	random.seed()
	for center_path in center_paths[:1]:
	    print 'Generating patches from {}'.format(center_path)
        weight_map = cv2.imread(center_path)
        img_name = os.path.split(center_path)[-1][:-len(weight_ext)]
        # read Image
        img_path = os.path.join(img_dir, img_name + img_ext)
        img = cv2.imread(img_path)
        # read dot annot
        annot_path = os.path.join(annot_dir, img_name + annot_ext)
        dot_annot = cv2.imread(annot_path)[:,:,1]
        imgH, imgW = img.shape[:2]
        #print sample_indice
        for iy, ix in zip(*np.nonzero(dot_annot)):
            left, top = ix-R, iy-R
            if left < 0 or top < 0: continue
            if left+patch_size > imgW or top+patch_size > imgH: continue
            patch_img = img[top:top+patch_size, left:left+patch_size,:]
            patch_weight = weight_map[top:top+patch_size, left:left+patch_size]
            for rotk in rot_angles: 
                patch_img_rot = np.rot90(patch_img, rotk)
                patch_weight_rot = np.rot90(patch_weight, rotk)
                pid = '{}_{}_{}'.format(iy, ix, rotk)
                patch_img_dst = os.path.join(train_dir, img_name + '_' + pid + '.png')
                cv2.imwrite(patch_img_dst, patch_img_rot)
                patch_weight_dst = os.path.join(train_dir, img_name + '_' + pid + '_weight.png')
                cv2.imwrite(patch_weight_dst, patch_weight_rot)

if __name__ == '__main__':
    if os.path.isdir(train_dir):
    	shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    generate_training_data_sampling()    