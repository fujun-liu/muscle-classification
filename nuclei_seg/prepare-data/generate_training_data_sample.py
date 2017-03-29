import os,sys
import glob,shutil
import numpy as np
import random
from PIL import Image, ImageDraw
#import cv2

img_dir = 'img-train'
center_mask_dir, fg_mask_dir, weight_map_dir = img_dir, img_dir, img_dir 
img_ext, center_mask_ext, fg_mask_ext, weight_map_ext, edge_ext = '.tif', '_center.png', '_fg.png', '_weight.png', '_edge.png'
patch_size = 128
stride = 32
train_ratio = 1.0
train_dir = 'train'
test_dir = 'test'
train_lst_file = 'train_pair.lst'
test_lst_file = 'test_pair.lst'
mean_file = 'mean.txt'

def generate_training_data():
	'''
	Generate training samples for training
	'''

	img_paths = glob.glob(img_dir + '/*' + img_ext)
	all_mean = np.zeros(3)
	train_lst = []
	test_lst = []
	
	R = patch_size/2
	
	rot_angles = [0,1,2]
	random.seed()
	for img_path in img_paths:
		print 'Generating patches from {}'.format(img_path)
		# normalize data in the range [0,1], case image max valus exceeds 255
		#img = 1.0*cv2.imread(img_path)
		img = 1.0*np.array(Image.open(img_path))
		img = img/max(255.0, 1.0*np.max(img))
		imgH, imgW, _ = img.shape
		all_mean += np.sum(np.sum(img,axis=0), axis=0)/(imgH*imgW)
		# read center mask and foreground mask
		img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
		#img_center_mask = cv2.imread(os.path.join(center_mask_dir, img_name + center_mask_ext))
		img_center_mask = np.array(Image.open(os.path.join(center_mask_dir, img_name + center_mask_ext)))
		if img_center_mask is None:
			print 'Invalid center mask path {}'.format(os.path.join(center_mask_dir, img_name + center_mask_ext))
			sys.exit(1)
		#img_fg_mask = cv2.imread(os.path.join(fg_mask_dir, img_name + fg_mask_ext))
		img_fg_mask = np.array(Image.open(os.path.join(fg_mask_dir, img_name + fg_mask_ext)))
		if img_fg_mask is None:
			print 'Invalid fg mask path {}'.format(os.path.join(fg_mask_dir, img_name + fg_mask_ext))
			sys.exit(1)
		img_edge_mask = np.array(Image.open(os.path.join(fg_mask_dir, img_name + edge_ext)))
		
		weight_map = np.array(Image.open(os.path.join(weight_map_dir, img_name + weight_map_ext)))
		# get all patch coords
		start_x = R + random.randint(0, stride-1)
		start_y = R + random.randint(0, stride-1)
		gridX, gridY = range(start_x,imgW-R,stride), range(start_y,imgH-R,stride)

		if len(gridX) == 0 or len(gridY) == 0:
			continue
		[meshY, meshX] = np.meshgrid(gridY, gridX)
		meshY = meshY.flatten()
		meshX = meshX.flatten()

		#print sample_indice
		for iy, ix in zip(meshY, meshX):
			patch_img = img[iy-R+1:iy+R+1, ix-R+1:ix+R+1,:]
			patch_center = img_center_mask[iy-R+1:iy+R+1, ix-R+1:ix+R+1]
			patch_fg = img_fg_mask[iy-R+1:iy+R+1, ix-R+1:ix+R+1]
			patch_edge = img_edge_mask[iy-R+1:iy+R+1, ix-R+1:ix+R+1]
			patch_weight = weight_map[iy-R+1:iy+R+1, ix-R+1:ix+R+1]
			for rotk in rot_angles: 
				patch_img_rot = np.rot90(patch_img, rotk)
				patch_center_rot = np.rot90(patch_center, rotk)
				patch_fg_rot = np.rot90(patch_fg, rotk)
				patch_edge_rot = np.rot90(patch_edge, rotk)
				patch_weight_rot = np.rot90(patch_weight, rotk)
				pid = '{}_{}_{}'.format(iy, ix, rotk)
				if np.random.random() < train_ratio:
					# for training
					patch_img_dst = os.path.join(train_dir, img_name + '_' + pid + '.png')
					patch_center_dst = os.path.join(train_dir, img_name + '_' + pid + '_center.png')
					patch_fg_dst = os.path.join(train_dir, img_name + '_' + pid + '_fg.png')
					patch_edge_dst = os.path.join(train_dir, img_name + '_' + pid + '_edge.png')
					patch_weight_dst = os.path.join(train_dir, img_name + '_' + pid + '_weight.png')
					train_lst.append((patch_img_dst, patch_center_dst,patch_fg_dst, patch_weight_dst, patch_edge_dst))
				else:
					# for testing
					patch_img_dst = os.path.join(test_dir, img_name + '_' + pid + '.png')
					patch_center_dst = os.path.join(test_dir, img_name + '_' + pid + '_center.png')
					patch_fg_dst = os.path.join(test_dir, img_name + '_' + pid + '_fg.png')
					patch_edge_dst = os.path.join(test_dir, img_name + '_' + pid + '_edge.png')
					patch_weight_dst = os.path.join(test_dir, img_name + '_' + pid + '_weight.png')
					test_lst.append((patch_img_dst, patch_center_dst, patch_fg_dst, patch_weight_dst, patch_edge_dst))
				# save data into its folder
				#cv2.imwrite(patch_img_dst, 255*patch_img_rot)	
				Image.fromarray((255*patch_img_rot).astype(np.uint8)).save(patch_img_dst, 'png')
				#cv2.imwrite(patch_center_dst, patch_center_rot)	
				Image.fromarray((patch_center_rot).astype(np.uint8)).save(patch_center_dst, 'png')
				#cv2.imwrite(patch_fg_dst, patch_fg_rot)	
				Image.fromarray((patch_fg_rot).astype(np.uint8)).save(patch_fg_dst, 'png')
				Image.fromarray((patch_edge_rot).astype(np.uint8)).save(patch_edge_dst, 'png')
				Image.fromarray((patch_weight_rot).astype(np.uint8)).save(patch_weight_dst, 'png')
	# write train list
	with open(train_lst_file, 'w') as f:
		for img_path, center_path, fg_path, weight_path, edge_path in train_lst:
			f.write('{} {} {} {} {}\n'.format(img_path, center_path, fg_path, edge_path, weight_path))
	# write test lst
	with open(test_lst_file, 'w') as f:
		for img_path, center_path, fg_path, weight_path, edge_path in test_lst:
			f.write('{} {} {} {} {}\n'.format(img_path, center_path, fg_path, edge_path, weight_path))
	# write mean to file
	all_mean /= len(img_paths)
	with open(mean_file, 'w') as f:
		f.write('{} {} {}\n'.format(all_mean[0], all_mean[1], all_mean[2]))


if __name__ == '__main__':
	if os.path.isdir(train_dir):
		shutil.rmtree(train_dir)
	if os.path.isdir(test_dir):
		shutil.rmtree(test_dir)
	os.mkdir(train_dir)
	os.mkdir(test_dir)
	generate_training_data()
