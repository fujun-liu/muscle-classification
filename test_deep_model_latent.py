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

arch = 'resnet'
use_l1 = False
use_enhance = False
use_last = False
use_raw = False
use_cv_datasets = False
use_tile = False
use_em = False
use_latent = True
batch_sz = 10

crop_size = 512
train_size = 256
R = crop_size // 2
#model_dir = 'torch-model'
model_dir = 'torch-model'
result_dir = 'deep-feat'

save_result = False
em_topk = 30
# number of rois used for training
train_num_rois = 30
# number of rois used for testing
test_num_rois = 30
if not use_tile:
    if use_em: result_name = 'deep-feat-{}-{}-em-is'.format(arch, test_num_rois)
    else: 
        result_name = 'deep-feat-{}-train{}-test{}'.format(arch, train_num_rois, test_num_rois)
        if use_l1: result_name += '-l1'
        if use_enhance: result_name += '-enhance'
        if use_last: result_name += '-last'
        if use_raw: result_name += '-raw'
        if use_cv_datasets: result_name += '-cv-datasets'
        if use_latent: result_name += '-latent'
else:
    if use_em:  result_name = 'deep-feat-{}-tile-200-em{}'.format(arch, em_topk)
    else: result_name = 'deep-feat-{}-tile'.format(arch)
# load cached roi centers
#roi_center_path = None
roi_center_path = 'slide-feat/roi_centers_{}.p'.format(test_num_rois)
if roi_center_path is not None and not os.path.exists(roi_center_path):
    roi_center_path = None
if use_tile: roi_center_path = None

nuclei_dir = '/media/fujunl/FujunLiu/muscle-classification/muscle-whole-slide-results/nuclei-det'
nuclei_suffix = '_nuclei_seg.png'
fg_dir = '/media/fujunl/FujunLiu/muscle-classification/muscle-whole-slide-results/cell-seg'
fg_suffix = '_fg_noholes.png'

wsi_dir = '/media/fujunl/FujunLiu/muscle-classification/wholeslide/clean'
if use_cv_datasets:
    cv_split_info_file = 'cv2_split_info.p'
    num_classes = 2
else:
    cv_split_info_file = 'cv3_split_info.p'
    num_classes = 3

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
avg_voting_acc_latent = .0
avg_roi_acc = .0

roi_center_dict = None
if roi_center_path is not None:
    with open(roi_center_path, 'rb') as f:
        roi_center_dict = pickle.load(f)

patient_info = {}

for gid in range(num_folds):
    print 'Testing set {}'.format(gid)
    if not use_tile:
        if not use_em: 
            model_name = 'pretrain_{}classes_cv{}{}_{}_60_{}'.format(num_folds, num_folds, gid, arch, train_num_rois)
            if use_l1: model_name += '_l1_only'
            if use_enhance: model_name += '_enhance'
            if use_last: model_name += '_last'
            if use_raw: model_name += '_raw'
            if use_cv_datasets: model_name += '_cv_datasets'
            if use_latent: model_name += '_latent'
        else: model_name = 'pretrain_{}classes_cv{}{}_{}_60_em_3010_is'.format(num_folds, num_folds, gid, arch)
    else:
        if use_em: model_name = 'pretrain_{}classes_cv{}{}_{}_200_em_tile{}'.format(num_folds, num_folds, gid, arch, em_topk)
        else: model_name = 'pretrain_{}classes_cv{}{}_{}_20_tiles'.format(num_folds, num_folds, gid, arch)
    #model_name = 'pretrain_3classes_cv3{}_resnetup8_60_10_test'.format(gid)
    model_path = os.path.join(model_dir, model_name + '.th')
    print model_path
    #if not os.path.exists(model_path):
    #    tmp_num_rois = 10
    #    model_name = 'pretrain_{}classes_cv{}{}_{}_60_{}'.format(num_folds, num_folds, gid, arch, tmp_num_rois)
    #    model_path = os.path.join(model_dir, model_name + '.th')
    muscle_model = MuscleModelPredict(num_classes=num_classes)
    feat_extractor = MuscleFeatExtractor(num_classes=num_classes)
    muscle_model.init_test(model_path, test_transform=test_transform, batch_sz=batch_sz)
    feat_extractor.init_extract(model_path, arch, test_transform=test_transform, batch_sz=batch_sz)
    # load deep learning model
    slide_lst = []
    for slide_name in wsi_name_lst:
        label, pid = label_parser(slide_name)
        if cv_split_info[pid] == gid:
            slide_lst.append(slide_name)
    voting_acc = .0
    voting_acc_latent = .0
    roi_acc = .0
    roi_cnt = .0
    for slide_name in slide_lst:
        label, pid = label_parser(slide_name)
        if use_cv_datasets: label = 1 if label == 0 else 0
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
        for ri in range(roi_centers.shape[0]):
            y, x = roi_centers[ri]
            top, bottom = y-R, y+R
            left, right = x-R, x+R
            if top < 0 or bottom > H or left < 0 or right > W:
                continue
            roi = img[top:bottom, left:right]
            #img_lst.append(Image.fromarray(roi))
            img_lst.append(Image.fromarray(cv2.resize(roi, (train_size, train_size))))
        proba, proba_latent = muscle_model.pred_proba_latent(img_lst)
        deep_feat = feat_extractor.extract(img_lst)
        print deep_feat.shape
        print proba.shape, proba_latent.shape

        if pid in patient_info.keys():
            assert patient_info[pid]['gid'] == gid
            assert patient_info[pid]['label'] == label
            patient_info[pid]['slides'].append(slide_name)
            patient_info[pid]['proba'] = np.concatenate((patient_info[pid]['proba'], proba), axis=0)
            patient_info[pid]['proba_latent'] = np.concatenate((patient_info[pid]['proba_latent'], proba_latent), axis=0)
            patient_info[pid]['feat'] = np.concatenate((patient_info[pid]['feat'], deep_feat), axis=0)
        else:
            patient_info[pid] = {'proba':proba, 'proba_latent':proba_latent, 'gid':gid, 'label':label, 'slides':[slide_name], 'feat':deep_feat}
        pred_slide = np.argmax(np.mean(proba, axis=0))
        pred_slide_latent = np.argmax(np.mean(proba_latent[:,:3], axis=0))
        pred_roi = np.argmax(proba, axis=1)
        voting_acc += pred_slide == label
        voting_acc_latent += pred_slide_latent == label

        roi_acc += np.sum(pred_roi == label)
        roi_cnt += proba.shape[0]
    roi_acc /= roi_cnt
    voting_acc /= len(slide_lst)
    voting_acc_latent /= len(slide_lst)
    avg_voting_acc += voting_acc
    avg_voting_acc_latent += voting_acc_latent
    avg_roi_acc += roi_acc
    print 'roi acc: {}'.format(roi_acc)
    print 'voting acc: {}'.format(voting_acc)
    print 'latent voting acc: {}'.format(voting_acc_latent)
    
print 'avg roi acc: {}'.format(avg_roi_acc/num_folds)
print 'avg voting acc: {}'.format(avg_voting_acc/num_folds)
print 'avg latent voting acc: {}'.format(avg_voting_acc_latent/num_folds)
# test patient classification acc
patien_acc = .0
patient_latent_acc = .0
conf_mat = np.zeros((num_classes,num_classes))
conf_mat_latent = np.zeros((num_classes,num_classes))
for pid, pid_proba in patient_info.iteritems():
    proba = pid_proba['proba']
    proba_latent = pid_proba['proba_latent'][:,:3]
    label = pid_proba['label']
    pred = np.argmax(np.mean(proba, axis=0))
    pred_latent = np.argmax(np.mean(proba_latent, axis=0))
    patien_acc += pred == label
    patient_latent_acc += pred_latent == label
    conf_mat[label, pred] += 1.0
    conf_mat[label, pred_latent] += 1.0
print 'Patient acc: {}'.format(patien_acc/len(patient_info))
print conf_mat
print 'Patient latent acc: {}'.format(patient_latent_acc/len(patient_info))
print conf_mat_latent

if save_result:
    # save to pickle
    result_path = os.path.join(result_dir, result_name + '.p')
    with open(result_path, 'wb') as f:
        pickle.dump(patient_info, f)
    # save to mat
    mat_path = os.path.join(result_dir, 'mat', result_name + '.mat')
    savemat(mat_path, mdict=patient_info)