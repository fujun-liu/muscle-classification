'''
crop rois
'''
from torchvision import transforms
import glob, os, pickle
from class_label_parser import LabelParser
from util import get_dense_nuclei_rois, get_tile_rois
from MuscleModelPredict import MuscleModelPredict, data_transforms
from MuscleModelFeature import MuscleFeatExtractor
import cv2
import numpy as np
from PIL import Image
from scipy.io import savemat
from ResNetVisulize import ResNetVisulize
from skimage.transform import pyramid_expand

arch = 'resnet'
use_l1 = False
atten_type = 'cam'
batch_sz = 10

crop_size = 512
train_size = 256
R = crop_size // 2
#model_dir = 'torch-model'
model_dir = 'torch-model'
result_dir = '/media/fujunl/FujunLiu/muscle-classification/atten-whole'

# number of rois used for training
train_num_rois = 30
# number of rois used for testing
test_num_rois = 30

# load cached roi centers
#roi_center_path = None
roi_center_path = 'slide-feat/roi_centers_{}.p'.format(test_num_rois)
if roi_center_path is not None and not os.path.exists(roi_center_path):
    roi_center_path = None


nuclei_dir = '/media/fujunl/FujunLiu/muscle-classification/muscle-whole-slide-results/nuclei-det'
nuclei_suffix = '_nuclei_seg.png'
fg_dir = '/media/fujunl/FujunLiu/muscle-classification/muscle-whole-slide-results/cell-seg'
fg_suffix = '_fg_noholes.png'

wsi_dir = '/media/fujunl/FujunLiu/muscle-classification/wholeslide/clean'
cv_split_info_file = 'cv3_split_info.p'

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

if 'inception' in arch:
    train_size = 320
    test_transform = transforms.Compose([
            transforms.Scale(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.6900, 0.3519, 0.5292], [0.1426,0.1989,0.1625])
    ])
else:
    test_transform = data_transforms['val']

num_folds = np.max(np.array(cv_split_info.values())) + 1
avg_voting_acc = .0
avg_roi_acc = .0

roi_center_dict = None
if roi_center_path is not None:
    with open(roi_center_path, 'rb') as f:
        roi_center_dict = pickle.load(f)

patient_info = {}

for gid in range(num_folds):
    print 'Testing set {}'.format(gid)
    model_name = 'pretrain_{}classes_cv{}{}_{}_60_{}'.format(num_folds, num_folds, gid, arch, train_num_rois)
    if use_l1:
        model_name += '_l1'
    #model_name = 'pretrain_3classes_cv3{}_resnetup8_60_10_test'.format(gid)
    model_path = os.path.join(model_dir, model_name + '.th')
    print model_path

    muscle_model = ResNetVisulize()
    muscle_model.init_atten(model_path, arch, atten_type=atten_type, test_transform=test_transform, batch_sz=batch_sz)
    # load deep learning model
    slide_lst = []
    for slide_name in wsi_name_lst:
        label, pid = label_parser(slide_name)
        if cv_split_info[pid] == gid:
            slide_lst.append(slide_name)
    voting_acc = .0
    roi_acc = .0
    roi_cnt = .0
    for slide_name in slide_lst:
        label, pid = label_parser(slide_name)
        # read nuclei map
        img_path = os.path.join(nuclei_dir, slide_name + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nuclei_path = os.path.join(nuclei_dir, slide_name + nuclei_suffix)
        nuclei_map = cv2.imread(nuclei_path)/255.0
        if len(nuclei_map.shape) == 3:
            nuclei_map = nuclei_map[:,:,0]
        H, W = nuclei_map.shape
        if roi_center_dict is not None:
            roi_centers = roi_center_dict[slide_name]
        else:
            # read fg
            fg_path = os.path.join(fg_dir, slide_name + fg_suffix)
            fg = cv2.resize(cv2.imread(fg_path), (W, H))
            if len(fg.shape) == 3:
                fg = fg[:,:,0]
            fg = fg > 0
            if not use_tile:
                roi_centers, weight = get_dense_nuclei_rois(nuclei_map, fg, test_num_rois, crop_size=crop_size)
            else:
                roi_centers = get_tile_rois(fg, tile_size=crop_size)
        img_lst = []
        valid_roi_centers = []
        for ri in range(roi_centers.shape[0]):
            y, x = roi_centers[ri]
            top, bottom = y-R, y+R
            left, right = x-R, x+R
            if top < 0 or bottom > H or left < 0 or right > W:
                continue
            roi = img[top:bottom, left:right]
            valid_roi_centers.append(roi_centers[ri])
            #img_lst.append(Image.fromarray(roi))
            img_lst.append(Image.fromarray(cv2.resize(roi, (train_size, train_size))))
        crop_lst, atten_lst, proba, act_max_val = muscle_model.compute_attention(img_lst)
        #slide_dir = os.path.join(result_dir, slide_name)
        #print slide_dir
        #print len(crop_lst)
        #if not os.path.isdir(slide_dir):
        #    os.makedirs(slide_dir)
        upscale = 2.0*crop_lst[0].shape[0]/atten_lst[0].shape[0]
        atten_map_whole = np.zeros((H,W))
        for ri in range(len(crop_lst)):
            pred = np.argmax(proba[ri])
            crop_path = '{}/{}-{}.png'.format(result_dir, slide_name, ri)
            atten_path = '{}/{}-{}-{}-T{}P{}-atten-{}'.format(result_dir, slide_name, ri, arch, label, pred, atten_type)
            if use_l1: atten_path += '-l1'
            crop_patch = cv2.resize(crop_lst[ri], (448,448))
            cv2.imwrite(crop_path, crop_patch)
            atten = atten_lst[ri]/act_max_val
            atten_up = pyramid_expand(atten, upscale=upscale)
            y, x = valid_roi_centers[ri]
            atten_h, atten_w = atten_up.shape
            top = y - atten_h // 2
            left = x - atten_w // 2
            atten_map_whole[top:top+atten_h, left:left+atten_w] = atten_up
            cv2.imwrite(atten_path + '.png', 255*atten_up)

        if pid in patient_info.keys():
            assert patient_info[pid]['gid'] == gid
            assert patient_info[pid]['label'] == label
            patient_info[pid]['slides'].append(slide_name)
            patient_info[pid]['proba'] = np.concatenate((patient_info[pid]['proba'], proba), axis=0)
        else:
            patient_info[pid] = {'proba':proba, 'gid':gid, 'label':label, 'slides':[slide_name]}
        pred_slide = np.argmax(np.mean(proba, axis=0))
        pred_roi = np.argmax(proba, axis=1)
        voting_acc += pred_slide == label
        roi_acc += np.sum(pred_roi == label)
        roi_cnt += proba.shape[0]

         # save whole atten map
        atten_path_whole = '{}/whole/{}-atten-{}-t{}p{}'.format(result_dir, slide_name, atten_type, label, pred_slide)
        if use_l1: atten_path_whole += '-l1'
        cv2.imwrite(atten_path_whole + '.png', 255*atten_map_whole)
        roi_centers_mat_path = '{}/whole/{}-roi-centers-{}.mat'.format(result_dir, slide_name, test_num_rois)
        savemat(roi_centers_mat_path, mdict={'centers':valid_roi_centers})

    roi_acc /= roi_cnt
    voting_acc /= len(slide_lst)
    avg_voting_acc += voting_acc
    avg_roi_acc += roi_acc
    print 'roi acc: {}'.format(roi_acc)
    print 'voting acc: {}'.format(voting_acc)
    
print 'avg roi acc: {}'.format(avg_roi_acc/num_folds)
print 'avg voting acc: {}'.format(avg_voting_acc/num_folds)
# test patient classification acc
patien_acc = .0
conf_mat = np.zeros((3,3))
for pid, pid_proba in patient_info.iteritems():
    proba = pid_proba['proba']
    label = pid_proba['label']
    pred = np.argmax(np.mean(proba, axis=0))
    patien_acc += pred == label
    conf_mat[label, pred] += 1.0
print 'Patient acc: {}'.format(patien_acc/len(patient_info))
print conf_mat