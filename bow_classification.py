import os, pickle
import numpy as np
from classification_utils import train_model_with_grid_search
from encoding_image_feature import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
'''
BoW classification
1. pca; 2. kmeans; 3. bow encoding of patients; 4. classification
'''
'''
format
feat_info = {'X_lst': X_lst, 'Y':Y, 'pid_all':pid_all, 'slide_name_all': slide_name_all, 
                'centers_lst':centers_lst, 'group_ids':group_ids, 'model_scores':model_scores, 'deep_feat':DeepFeat}
'''
pca_dim = 32
num_clusters = 16

def cv_bow(patient_info, pca_dim, num_clusters, num_folds=3,):
    '''
    format
    patient_info[pid] = {'proba':proba[i], 'gid':group_ids[i], 'label':Y[i], 
                                'slides':[slide_name_all[i]], 'feat':deep_feat[i]} 
    '''
    avg_voting_acc = .0
    avg_bow_acc = .0
    avg_ensemble_acc = .0
    for gid in range(num_folds):
        proba_train_lst, proba_test_lst = [], []
        X_train_lst, X_test_lst = [], []
        Y_train, Y_test = [], []
        deep_acc = .0
        for pid in patient_info.keys():
            if patient_info[pid]['gid'] == gid:
                deep_acc += patient_info[pid]['label'] == np.argmax(np.mean(patient_info[pid]['proba'], axis=0))
                # test data
                X_test_lst.append(patient_info[pid]['feat'])
                proba_test_lst.append(np.mean(patient_info[pid]['proba'], axis=0)[np.newaxis,:])
                Y_test.append(patient_info[pid]['label'])
            else:
                # train data
                X_train_lst.append(patient_info[pid]['feat'])
                proba_train_lst.append(np.mean(patient_info[pid]['proba'], axis=0)[np.newaxis,:])
                Y_train.append(patient_info[pid]['label'])
        print len(X_train_lst), len(X_test_lst)
        X_train = np.concatenate(X_train_lst, axis=0)
        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)
        # pca dim
        pca_model = PCA(n_components=pca_dim)
        X_train = pca_model.fit_transform(X_train)
        # l2 normalize again
        X_train = normalize_feature(X_train)
        # run kmeans on training
        kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
        kmeans_model.fit(X_train)
        centers = kmeans_model.cluster_centers_
        num_train, num_test = len(X_train_lst), len(X_test_lst)
        code_train = np.zeros((num_train, num_clusters))
        code_test = np.zeros((num_test, num_clusters))
        # encode train
        for i in range(num_train):
            after_pca = pca_model.transform(X_train_lst[i])
            code_train[i] = bow_encoding(centers, after_pca)
        # encode test
        for i in range(num_test):
            after_pca = pca_model.transform(X_test_lst[i])
            code_test[i] = bow_encoding(centers, after_pca)
        
        prob_train = np.concatenate(proba_train_lst, axis=0)
        prob_test = np.concatenate(proba_test_lst, axis=0)
        print code_train.shape, Y_train.shape
        print code_test.shape, Y_test.shape
        print prob_train.shape, prob_test.shape
        model, bow_acc = train_model_with_grid_search(code_train, Y_train, ('logit',), X_test=code_test, y_test=Y_test)
        bow_proba = model['pred_model'].predict_proba(model['norm_model'].transform(code_test))
        ensemble_pred = np.argmax(prob_test+bow_proba, axis=1)
        ensemble_acc = 1.0*np.sum(ensemble_pred == Y_test)/num_test
        voting_acc = deep_acc/num_test
        avg_bow_acc += bow_acc
        avg_voting_acc += voting_acc
        avg_ensemble_acc += ensemble_acc
        print 'voting acc: {}'.format(voting_acc)
        #print 'ensemble acc: {}'.format(ensemble_acc)
    
    print 'Avg bow acc: {}'.format(avg_bow_acc/num_folds)
    print 'Avg voting acc {}'.format(avg_voting_acc/num_folds)
    #print 'Avg ensemble acc {}'.format(avg_ensemble_acc/num_folds)


if __name__ == "__main__":
    feat_dir = 'deep-feat'
    feat_name = 'deep-feat-resnet-100'
    feat_path = os.path.join(feat_dir, feat_name + '.p')
    print feat_path
    #patient_info = load_old_patient_info(feat_path)
    with open(feat_path, 'rb') as f:
        patient_info = pickle.load(f)
    cv_bow(patient_info, pca_dim, num_clusters)

