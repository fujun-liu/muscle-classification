import os, pickle
import numpy as np
from classification_utils import train_model_with_grid_search
from encoding_image_feature import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def cv_region_classification(patient_info, num_folds=3):
    '''
    format
    patient_info[pid] = {'proba':proba[i], 'gid':group_ids[i], 'label':Y[i], 
                                'slides':[slide_name_all[i]], 'feat':deep_feat[i]} 
    '''
    avg_deep_acc = .0
    avg_roi_acc = .0
    for gid in range(num_folds):
        proba_train_lst, proba_test_lst = [], []
        X_train_lst, X_test_lst = [], []
        Y_train, Y_test = [], []
        deep_acc = .0
        for pid in patient_info.keys():
            feat = patient_info[pid]['feat']
            label = patient_info[pid]['label']
            proba = patient_info[pid]['proba']
            if patient_info[pid]['gid'] == gid:
                deep_acc += patient_info[pid]['label'] == np.argmax(np.mean(patient_info[pid]['proba'], axis=0))
                # test data
                X_test_lst.append(feat)
                proba_test_lst.append(proba)
                Y_test.extend([label]*feat.shape[0])
            else:
                # train data
                X_train_lst.append(feat)
                proba_train_lst.append(proba)
                Y_train.extend([label]*feat.shape[0])
        
        X_train = np.concatenate(X_train_lst, axis=0)
        X_test = np.concatenate(X_test_lst, axis=0)
        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)
        print X_train.shape, Y_train.shape
        print X_test.shape, Y_test.shape
        model, roi_acc = train_model_with_grid_search(X_train, Y_train, ('logit',), X_test=X_test, y_test=Y_test)
        prob_test = np.concatenate(proba_test_lst, axis=0)
        print prob_test.shape, Y_test.shape
        deep_pred = np.argmax(prob_test, axis=1)
        deep_acc = 1.0*np.sum(deep_pred == Y_test)/Y_test.size
        avg_deep_acc += deep_acc
        avg_roi_acc += roi_acc
        print 'deep acc: {}'.format(deep_acc)
        print 'roi acc: {}'.format(roi_acc)
    
    print 'Avg roi acc: {}'.format(avg_roi_acc/num_folds)
    print 'Avg deep acc {}'.format(avg_deep_acc/num_folds)
    

if __name__ == "__main__":
    feat_dir = 'deep-feat'
    feat_name = 'deep-feat-resnet-10'
    feat_path = os.path.join(feat_dir, feat_name + '.p')
    print feat_path
    #patient_info = load_old_patient_info(feat_path)
    with open(feat_path, 'rb') as f:
        patient_info = pickle.load(f)
    cv_region_classification(patient_info)

