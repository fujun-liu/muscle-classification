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

arch = 'resnetfinefine4'
batch_sz = 256

crop_size = 512
train_size = 256
R = crop_size // 2
#model_dir = 'torch-model'
model_dir = 'torch-model'
result_dir = 'whole-atten'

# number of rois used for training
train_num_rois = 10
overlap = 32
grid_sz = crop_size - overlap

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

patient_info = {}

for gid in range(num_folds):
    print 'Testing set {}'.format(gid)
    model_name = 'pretrain_{}classes_cv{}{}_{}_60_{}'.format(num_folds, num_folds, gid, arch, train_num_rois)
    #model_name = 'pretrain_3classes_cv3{}_resnetup8_60_10_test'.format(gid)
    model_path = os.path.join(model_dir, model_name + '.th')
    print model_path

    muscle_model = ResNetVisulize()
    muscle_model.init_atten(model_path, arch, test_transform=test_transform, batch_sz=batch_sz)
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
        nH, nW = int(np.round(1.0*H/grid_sz)), int(np.round(1.0*W/grid_sz))
        img_lst = []
        for i in range(nH):
            top = i * grid_sz
            bottom = top+crop_size if i != nH-1 else H
            for j in range(nW):
                left = j*grid_sz
                right = left+crop_size if j != nW-1 else W
                roi = img[top:bottom, left:right]
                img_lst.append(Image.fromarray(cv2.resize(roi, (train_size, train_size))))
        crop_lst, atten_lst, proba, maxval = muscle_model.compute_attention(img_lst)
        upscale = 1.0*crop_lst[0].shape[0]/atten_lst[0].shape[0]
        img_whole, atten_whole = None, None
        for i in range(nH):
            img_row, atten_row = None, None
            for j in range(nW):
                lin_ind = i*nW + j
                crop = crop_lst[lin_ind]
                atten = pyramid_expand(atten_lst[lin_ind]/maxval, upscale=upscale)
                img_row = crop if img_row is None else np.concatenate((img_row,crop), axis=1)
                atten_row = atten if atten_row is None else np.concatenate((atten_row,atten), axis=1)
            img_whole = img_row if img_whole is None else np.concatenate((img_whole,img_row), axis=0)
            atten_whole = atten_row if atten_whole is None else np.concatenate((atten_whole,atten_row), axis=0)
        whole_img_path = '{}/{}.png'.format(result_dir, slide_name)
        whole_atten_path = '{}/{}-{}-atten.png'.format(result_dir, slide_name, arch)
        cv2.imwrite(whole_img_path, img_whole)
        cv2.imwrite(whole_atten_path, atten_whole*255)
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

result_path = 'whole-info/{}-{}-whole.p'.format(arch, train_num_rois)
with open(result_path, 'wb') as f:
    pickle.dump(patient_info, f)