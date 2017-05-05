'''
crop rois
'''
import glob, os, pickle
from class_label_parser import LabelParser
from util import get_dense_nuclei_rois, get_tile_rois
import cv2
import numpy as np

use_tiles = False
num_rois = 10
crop_size = 512
train_size = 256
R = crop_size // 2
translate_size = 64

if use_tiles:
    crop_dir = '/media/fujunl/FujunLiu/muscle-classification/tiles'
else:
    crop_dir = '/media/fujunl/FujunLiu/muscle-classification/train-patches-{}-{}-enhance'.format(crop_size, num_rois)
if not os.path.isdir(crop_dir):
    os.makedirs(crop_dir)

wsi_dir = '/media/fujunl/FujunLiu/muscle-classification/wholeslide/clean'
cv_split_info_file = 'cv3_split_info.p'

nuclei_dir = '/media/fujunl/FujunLiu/muscle-classification/muscle-whole-slide-results/nuclei-det'
nuclei_suffix = '_nuclei_seg.png'
fg_dir = '/media/fujunl/FujunLiu/muscle-classification/muscle-whole-slide-results/cell-seg'
fg_suffix = '_fg_noholes.png'


# load wsi 
wsi_suffix = ('.ndpi', '.tiff', '.svs')
wsi_lst = []
wsi_name_lst = []
for suffix in wsi_suffix:
    tmp_lst = glob.glob(wsi_dir + '/*' + suffix)
    wsi_lst += tmp_lst
    wsi_name_lst += [os.path.split(path)[-1][:-len(suffix)] for path in tmp_lst]

with open(cv_split_info_file, 'rb') as f:
    cv_split_info = pickle.load(f)
label_parser = LabelParser()

tmp_label_cnt = np.zeros(3)
tmp_cv_cnt = np.zeros(3)
for slide_name in wsi_name_lst:
    print 'Extracting patches from {}'.format(slide_name)
    label, pid = label_parser(slide_name)
    gid = cv_split_info[pid]
    # read nuclei map
    img_path = os.path.join(nuclei_dir, slide_name + '.png')
    img = cv2.imread(img_path)
    nuclei_path = os.path.join(nuclei_dir, slide_name + nuclei_suffix)
    nuclei_map = cv2.imread(nuclei_path)/255.0
    if len(nuclei_map.shape) == 3:
        nuclei_map = nuclei_map[:,:,0]
    H, W = nuclei_map.shape
    # read fg
    fg_path = os.path.join(fg_dir, slide_name + fg_suffix)
    fg = cv2.resize(cv2.imread(fg_path), (W, H))
    if len(fg.shape) == 3:
        fg = fg[:,:,0]
    fg = fg > 0
    if use_tiles:
        roi_centers = get_tile_rois(fg, tile_size=crop_size)
        print roi_centers
    else:
        roi_centers, weight = get_dense_nuclei_rois(nuclei_map, fg, num_rois, crop_size=crop_size)
        print roi_centers, weight
    if translate_size > 0:
        up_shift = np.copy(roi_centers)
        up_shift[:,0] -= translate_size
        down_shift = np.copy(roi_centers)
        down_shift[:,0] += translate_size

        left_shift = np.copy(roi_centers)
        left_shift[:,1] -= translate_size
        right_shift = np.copy(roi_centers)
        right_shift[:,1] += translate_size
        roi_centers = np.concatenate((roi_centers, up_shift, down_shift, left_shift, right_shift), axis=0)

    for ri in range(roi_centers.shape[0]):
        y, x = roi_centers[ri]
        top, bottom = y-R, y+R
        left, right = x-R, x+R
        if top < 0 or bottom > H or left < 0 or right > W:
            continue
        roi_name = '{}-{}-{}-{}-{}'.format(slide_name, ri, crop_size, gid, label)
        roi_path = os.path.join(crop_dir, roi_name + '.png')
        roi = cv2.resize(img[top:bottom, left:right], (train_size, train_size))
        cv2.imwrite(roi_path, roi)
    tmp_label_cnt[label] += roi_centers.shape[0]
    tmp_cv_cnt[gid] += roi_centers.shape[0]

print tmp_label_cnt
print tmp_cv_cnt
    
