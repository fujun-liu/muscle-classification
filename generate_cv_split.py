'''
generate cv2 split
'''
import pickle
import glob
import os
import shutil
from class_label_parser import LabelParser
import numpy as np

cv3_split_file = 'cv3_split_info.p'
cv_dataset_split_file = 'cv2_split_info.p'
with open(cv3_split_file, 'rb') as f:
    cv3_split_info = pickle.load(f)

cv2_split_info = {}
cnt0, cnt1 = 0,0
for pid in cv3_split_info.keys():
    if pid.isdigit():
        cv2_split_info[pid] = 0
        cnt0 += 1
    else:
        cv2_split_info[pid] = 1
        cnt1 += 1
print cv2_split_info
print cnt0, cnt1
with open(cv_dataset_split_file, 'wb') as f:
    pickle.dump(cv2_split_info, f)

src_dir = '/media/fujunl/FujunLiu/muscle-classification/train-patches-10'
dst_dir = '/media/fujunl/FujunLiu/muscle-classification/train-patches-10-cv-datasets'
if not os.path.isdir(dst_dir):
    os.makedirs(dst_dir)

img_lst = glob.glob(src_dir + '/*.png')
label_parser = LabelParser()
label_cnt = np.zeros((2,3))
for img_path in img_lst:
    img_name = os.path.split(img_path)[-1][:-8]
    label, pid = label_parser(img_name)
    gid = cv2_split_info[pid]
    label_cnt[gid, label] += 1
    #dst_name = '{}_{}_{}'.format(img_name, gid, label)
    #shutil.copyfile(img_path, os.path.join(dst_dir, dst_name + '.png'))
print label_cnt