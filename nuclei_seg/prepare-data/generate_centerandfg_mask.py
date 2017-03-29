from scipy.io import loadmat
from scipy import ndimage
import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import cv2
import h5py
from skimage.measure import regionprops

def extimate_radius(contour):
	diameter = .0
	num_pts = contour.shape[1]
	for i in range(num_pts-1):
		for j in range(i+1, num_pts):
			diameter = max(diameter, np.linalg.norm(contour[:,i] - contour[:,j]))
	return diameter/2.0
			
def generate_masks(img_dir, img_ext, gt_dir, gt_ext, **kwargs):
	'''
	This function generate the contour and foreground masks, taking the image its contour points as input
	'''
	fixed_dis = 3.0
	radius_ratio = 0.5
	min_dis = 1.5
	sigma = 5
	img_paths = glob.glob(img_dir + '/*' + img_ext)
	for img_path in img_paths:
		print 'generate contour and foreground masks for {}'.format(img_path)
		img = np.asarray(Image.open(img_path))
		img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
		contour_path = gt_dir + '/' + img_name + gt_ext
		use_h5py = False
		try:
			xy = loadmat(contour_path)['Contours'][0]
		except NotImplementedError:
			# this is matlab -v7.3 file
			f = h5py.File(contour_path)
			xy = [f[element[0]][:] for element in f['Contours']]
			use_h5py = True
			print 'This is a -v7.3 file. Read it using h5py'
		imgH, imgW, _ = img.shape
		center_mask = np.zeros([imgH, imgW])
		fg_mask = np.zeros_like(center_mask)
		edge_mask = np.zeros_like(center_mask)
		weight_map = np.zeros_like(center_mask)
		for i in range(len(xy)):
			contour = xy[i]
			if use_h5py:
				contour = np.transpose(contour)
			xs = [min(x, imgW-1) for x in contour[0]]
			ys = [min(y, imgH-1) for y in contour[1]]
			
			# compute the polygon for this cell
			polygon = zip(xs, ys)
			cell_mask = Image.new('L', (imgW, imgH), 0)
			ImageDraw.Draw(cell_mask).polygon(polygon, fill=1)
			cell_mask = ndimage.binary_erosion(np.array(cell_mask, dtype=np.int), structure = np.ones((5,5)))
			fg_mask += np.array(cell_mask)
			edge_mask[ys, xs] = 1
			props = regionprops(cell_mask.astype(np.int))
			if fixed_dis > 0:
				dis_thresh = fixed_dis
			else:
				radius = 0.5 * props[0].minor_axis_length
				dis_thresh = max(min_dis, radius*radius_ratio)
			
			centers = props[0].centroid
			cell_center_mask = np.zeros([imgH, imgW])
			cell_center_mask[centers[0], centers[1]] = 1
			dist = ndimage.distance_transform_edt(1 - cell_center_mask)
			weight_map += np.exp(-dist**2/sigma**2)
			center_mask += dist < dis_thresh 
		weight_map[weight_map > 1.0] = 1.0
		fg_mask[fg_mask > 0] = 1	
		center_mask[center_mask > 0] = 1	
		
		edge_mask = cv2.dilate(edge_mask, np.ones((5,5),np.uint8), iterations=1)
		edge_mask = fg_mask * edge_mask	

		Image.fromarray((edge_mask*255).astype(np.uint8)).save(img_dir + '/' + img_name + '_edge.png', 'png')
		Image.fromarray((fg_mask*255).astype(np.uint8)).save(img_dir + '/' + img_name + '_fg.png', 'png')
		Image.fromarray((center_mask*255).astype(np.uint8)).save(img_dir + '/' + img_name + '_center.png', 'png')
		Image.fromarray((weight_map*255).astype(np.uint8)).save(img_dir + '/' + img_name + '_weight.png', 'png')

if __name__ == '__main__':
	
	img_dir = 'img-train'
	gt_dir = 'img-train'
	img_ext, gt_ext = '.tif', '_seg.mat'
	generate_masks(img_dir, img_ext, gt_dir, gt_ext)


